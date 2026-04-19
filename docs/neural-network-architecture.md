# Chess Value Network: Architecture & Design Decisions

## Overview

This project trains a neural network to evaluate chess positions — producing a scalar in
(−1, +1) representing the probability that White wins (positive) or Black wins (negative)
from a given position. The network is trained purely on game outcomes from PGN files and
self-play; it requires no external oracle like Stockfish.

---

## Why Not Stockfish?

Stockfish uses classical **alpha-beta search** with a hand-crafted evaluation function (NNUE
since v12). Its evaluation is fast and highly tuned, but:

- The heuristics (material balance, mobility, king safety weights) were designed by humans over
  decades and are difficult to extend or generalise.
- It couples evaluation tightly to search — the eval only makes sense in the context of the
  search tree.
- NNUE (`efficiently updatable neural network`) is a shallow linear network that compresses
  position features; it is not end-to-end learned from game outcomes.

Our network learns *all* its features from data. No domain-specific heuristics are hard-coded
beyond the input encoding.

---

## Why Not AlphaZero's Architecture?

AlphaZero (Silver et al., 2017) uses a **pure ResNet**: 20 residual blocks, each 256 filters,
feeding both a value head and a policy head. It is trained entirely via self-play with MCTS.

Differences from our approach:

| Aspect | AlphaZero | This project |
|---|---|---|
| Architecture | Pure CNN (ResNet ×20) | CNN + Transformer |
| Training signal | MCTS improved policy | Game outcomes only |
| Parameter count | ~22 M | ~8 M |
| Hardware | TPUs (5000+) | CPU / Metal GPU |
| Inference | Needs MCTS at inference | Single forward pass |
| Global reasoning | Residual skip connections only | Multi-head attention |

The pure ResNet is excellent at *local* pattern recognition (piece clusters, pawn chains) but
attention is weak across the board — two distant squares communicate only through many
stacked convolutions. A transformer handles this natively.

---

## Why Not a Pure Transformer?

A plain transformer (ViT-style, treating each square as a token) can reason globally but
lacks **spatial inductive bias**: the network must learn from scratch that adjacent squares
matter more than distant ones, that a knight attacks in an L-shape, etc. This requires much
more data and training time.

The CNN backbone provides these priors for free: convolutional kernels naturally detect local
structure, and the skip connections let the network compose them hierarchically.

---

## Our Approach: Hybrid CNN + Transformer

We combine the strengths of both:

```
Input [17 × 8 × 8]
    CNN backbone  → local features, spatial structure
    Transformer   → global board reasoning
    Value head    → scalar evaluation
```

This is the architecture used in modern Leela Chess Zero (LC0 v0.29+ "transformer nets").

---

## Input Encoding `[17 × 8 × 8]`

| Planes | Content |
|---|---|
| 0 – 5  | White pieces: pawn, knight, bishop, rook, queen, king (binary) |
| 6 – 11 | Black pieces: pawn, knight, bishop, rook, queen, king (binary) |
| 12     | Side to move (all 1.0 if White, all 0.0 if Black) |
| 13     | Castling right: White kingside (all 1.0 if available) |
| 14     | Castling right: White queenside |
| 15     | Castling right: Black kingside |
| 16     | Castling right: Black queenside |

Each square is a binary feature vector of depth 17. This is the same representation used in
AlphaZero, minus the move-history planes (we use 1 position history for simplicity).

**Why plane encodings instead of piece-list vectors?**
Convolutions operate spatially. The CNN sees rank/file structure directly. A flat piece-list
would destroy spatial relationships and require the model to re-learn board geometry.

---

## CNN Backbone: ResNet

```
Conv2d(17 → 256, 3×3, pad=1) → BatchNorm → ReLU
ResidualBlock ×8
    each: Conv→BN→ReLU→Conv→BN → Add(input) → ReLU
→ output: [256 × 8 × 8]
```

### Why residual blocks?

Without skip connections, deep CNNs suffer vanishing gradients: the signal from the loss
shrinks as it is multiplied through many layers during backprop. A residual block:

```
y = F(x) + x
```

means the gradient flows *directly* through the addition (dy/dx ≥ 1), keeping gradients
well-conditioned even with 8+ blocks.

Each block learns a *correction* to the identity — making training stable and letting
depth accumulate useful representations incrementally.

### Why 8 blocks?

AlphaZero used 20 blocks on TPUs with unlimited compute. We target CPU/Metal training in
reasonable time (~8 hours for 50k games). Empirically, 8 blocks vs. 4 blocks roughly halves
the validation loss at the same training time budget on the target hardware.

---

## Bridge: Reshape + CLS Token + Positional Embeddings

After the backbone, spatial feature maps `[256 × 8 × 8]` are reshaped to `[64, 256]` — each
of the 64 squares becomes one 256-dimensional **token**.

A learnable **CLS token** `[1, 256]` is prepended, giving sequence length 65. CLS is a
convention from BERT: a special token with no fixed meaning that, after attending to all
other tokens, serves as a global summary. The value head reads from CLS row only.

**Positional embeddings** `[65, 256]` (fully learned, one per sequence position) are added
elementwise. Without them, the transformer is permutation-invariant — it cannot tell apart
"knight on e4" from "knight on a1". With position embeddings, each square has a unique bias
added before attention.

---

## Transformer Encoder: 4 Blocks

```
TransformerBlock ×4 (d_model=256, num_heads=8, d_ff=1024)
    MultiHeadAttention → Add + LayerNorm
    FFN: Linear(256→1024, GELU) → Linear(1024→256) → Add + LayerNorm
```

### Multi-Head Attention

Attention `[S, S]` between 65 tokens lets the network compute:

- "Is my king safe given the opponent's queen position?"  
- "Does this knight control the centre?"  
- "Are both rooks connected?"

Each of 8 heads specialises in different relationships. With `d_k = 32` per head:

```
Attention(Q, K, V) = softmax(Q Kᵀ / √32) @ V
```

The `√32` scaling prevents softmax saturation for large dot products.

### Why GELU in the FFN, not ReLU?

GELU (Gaussian Error Linear Unit) smoothly gates input by its cumulative normal distribution:

```
GELU(x) ≈ 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))
```

It has been empirically shown to outperform ReLU in transformer FFNs (Hendrycks & Gimpel,
2016) because the soft gating prevents "dead neurons" and the smooth derivative helps
gradient flow.

### Why 4 transformer blocks?

Each block increases the "reasoning horizon" — after 4 blocks, every token has attended to
every other token 4 times via different projections. For chess on an 8×8 board with 65
tokens, 4 blocks is sufficient to propagate information across the whole board. More blocks
improve quality but increase training time quadratically in the number of layers.

---

## Value Head

```
CLS token [256] → Linear(256 → 1) → tanh → value ∈ (-1, +1)
```

**Why tanh?**

Game outcome labels are {-1, 0, +1}. tanh maps ℝ → (-1, +1) smoothly, matching the label
range. This prevents the linear head from growing unboundedly and avoids gradient explosion
during early training. Unlike sigmoid (0,1), tanh is zero-centered — important for the
MSE loss to be symmetric for White/Black positions.

**Why not a policy head?**

AlphaZero trains both a value head (who wins?) and a policy head (what move to make?),
combining them in MCTS. This is more powerful but requires significant training data and MCTS
at inference. Our goal is a fast position evaluator; the policy head and MCTS can be added
later on top of this foundation.

---

## Training

### Loss Function

Mean Squared Error between predicted value and game outcome:

```
loss = (1/N) Σ (predict(board_i) − outcome_i)²
```

The outcome is the terminal game result for the game in which position `i` occurred.

**Why MSE instead of cross-entropy?**

Cross-entropy requires one-hot targets (3 classes: win/draw/loss). MSE treats the output as a
continuous regression, which better fits tanh's continuous output. Draws (0.0) naturally fall
between wins (+1) and losses (-1) without discretisation.

### Optimizer: Adam

Adam (Kingma & Ba, 2014) maintains per-parameter first and second moment estimates:

```
m = β₁·m + (1−β₁)·g
v = β₂·v + (1−β₂)·g²
param -= lr · m̂ / (√v̂ + ε)
```

It adapts the effective learning rate per parameter, making it robust to sparse gradients
(rare piece positions) and asymmetric curvature (common in deep networks). Default
hyperparameters: `lr=1e-4, β₁=0.9, β₂=0.999, ε=1e-8`.

---

## Custom ML Framework (`tensor` crate)

Rather than using Burn or Candle, we built a minimal ML framework from scratch. Key reasons:

1. **Understanding**: Every layer, every gradient, every GPU dispatch is explicit and
   inspectable. There are no abstraction layers between you and the computation.
2. **Control**: Beam Metal (wgpu compiles to Metal on macOS) for GPU acceleration without
   depending on CUDA.
3. **Simplicity**: The entire framework is ~1500 lines of Rust. Burn is 150k+ lines.

### Autograd Design

Each tensor operation creates a computation graph node: a `GradFn` struct that holds clones
(cheap Arc copies) of its inputs and knows how to compute `∂loss/∂input` given
`∂loss/∂output`.

`backward()` performs a topological sort of the graph (DFS from the loss node), then iterates
in reverse order, calling each `GradFn::backward()`. Gradients accumulate into each tensor's
`.grad` field (sum over multiple uses).

This is the same design used by PyTorch's autograd engine, just with a simpler implementation
that matches our needs.

---

## Parameter Count (approximate)

| Component | Parameters |
|---|---|
| CNN backbone (conv_in + 8 res blocks) | ~5.0 M |
| CLS token | 256 |
| Positional embeddings | 16,640 |
| Transformer encoder (4 blocks) | ~2.1 M |
| Value head | 257 |
| **Total** | **~7.1 M** |

This is trainable to useful accuracy on a modern laptop CPU in a few hours with ~100k games.
A Metal GPU will be ~5–10× faster via the wgpu backend.

---

## Comparison Summary

| System | Architecture | Training | Inference | Strength |
|---|---|---|---|---|
| Stockfish NNUE | Shallow linear NN on hand-crafted features | Distillation from classical | With search | ~3500 Elo |
| AlphaZero | Deep ResNet (×20), policy + value | Pure self-play + MCTS | Requires MCTS | ~3500 Elo |
| LC0 (modern) | ResNet or Hybrid + Transformer | Self-play + MCTS | Requires MCTS | ~3500 Elo |
| **This project** | **Hybrid ResNet + Transformer (×8 + ×4)** | **PGN games + self-play, no MCTS** | **Single forward pass** | **Learning** |

The goal is not to match Stockfish but to build a complete, understandable, end-to-end
learned system where every component is transparent and hackable.
