# Chess Hybrid Value Network — Implementation Plan

## Context

We're building a chess position evaluator using a **hybrid CNN + Transformer architecture**
(the modern Leela Chess Zero approach): a ResNet backbone extracts local spatial features
(pawn structure, knight forks, piece clusters), then a Transformer reasons globally over those
features (king safety across the board, long-range tactics).

Output: a scalar ∈ (-1, +1) via tanh — trained on game outcomes (+1 white wins, 0 draw, -1
black wins) from PGN files and self-play. No Stockfish required.

Rather than using Burn or Candle, we build our own ML framework (`tensor` crate) from scratch:
full autograd (computation graph + backward pass), wgpu GPU backend (compiles to Metal on macOS),
NN layers (Conv2d, BatchNorm, LayerNorm, MultiHeadAttention), and Adam optimizer.

---

## Full Architecture

```
Input: [17 × 8 × 8]
  12 piece planes (binary: 1 where piece exists)
  + 1 side-to-move plane (all 1s = white, all 0s = black)
  + 4 castling planes (WK, WQ, BK, BQ — all 1s if right available)

  ↓ Conv2d(17→256, kernel=3×3, padding=1) → BatchNorm → ReLU   [256×8×8]

  ↓ ResidualBlock ×8                                            [256×8×8]
       each block:
         Conv2d(256→256, 3×3, pad=1) → BN → ReLU
         Conv2d(256→256, 3×3, pad=1) → BN
         Add(input) → ReLU

  ↓ reshape [256×8×8] → [64×256]   (each square = one 256-dim token)

  ↓ prepend learnable [CLS] token   [65×256]

  ↓ + positional embeddings P[65×256]  (learned)

  ↓ TransformerBlock ×4                                         [65×256]
       MultiHeadAttention (8 heads, d_k=32) → Add+Norm
       FFN (256→1024→256, GELU) → Add+Norm

  ↓ take row 0 (CLS)                                            [256]

  ↓ Linear(256→1) → tanh

  value ∈ (-1, +1)
```

**Parameter count (approx):** ~8M — trainable on CPU, fast on Metal.

---

## Crate Structure

```
crates/
  chess/     ← existing: Board, FEN, movegen (unchanged)
  tensor/    ← NEW: ML framework (autograd + wgpu + all layers + Adam)
  engine/    ← NEW: chess hybrid model + encoding + training + self-play
  cli/       ← existing: add train/eval/selfplay subcommands
  core/      ← existing (unchanged)
```

Dependency graph: `cli → engine → { chess, tensor }` and `tensor → { wgpu, ndarray }`

---

## Phase 1 — `tensor` crate: Storage & Tensor

**Files:**
- `crates/tensor/Cargo.toml` — deps: `wgpu`, `ndarray`, `pollster`, `bytemuck`
- `crates/tensor/src/lib.rs`
- `crates/tensor/src/storage.rs`
- `crates/tensor/src/tensor.rs`
- `crates/tensor/src/device.rs`

**`storage.rs`:**
```rust
pub enum Storage {
    Cpu(Vec<f32>),
    Gpu(Arc<wgpu::Buffer>),
}
```

**`device.rs`:** `GpuContext { device: wgpu::Device, queue: wgpu::Queue }` — initialized once.

**`tensor.rs`:**
```rust
pub struct Tensor {
    pub data:          Arc<Storage>,
    pub shape:         Vec<usize>,
    pub grad:          Option<Arc<Mutex<Tensor>>>,
    pub(crate) grad_fn: Option<Arc<dyn GradFn>>,
    pub requires_grad: bool,
}
impl Tensor {
    pub fn zeros(shape: &[usize], device: &Device) -> Self
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self
    pub fn to_gpu(&self, ctx: &GpuContext) -> Self
    pub fn to_cpu(&self) -> Self          // GPU readback
    pub fn backward(&self)                // triggers autograd
    pub fn t(&self) -> Self               // transpose (2D)
    pub fn reshape(&self, shape: &[usize]) -> Self
}
```

---

## Phase 2 — `tensor` crate: Autograd

**Files:**
- `crates/tensor/src/grad.rs`
- `crates/tensor/src/graph.rs`

**`grad.rs`:**
```rust
pub trait GradFn: Send + Sync {
    fn backward(&self, grad_output: &Tensor);
}
```

**`graph.rs` — `backward()` algorithm:**
1. Topological sort of DAG via DFS on `grad_fn` links
2. Seed: `loss.grad = Tensor::ones(loss.shape)`
3. Iterate reverse topological order → call `node.grad_fn.backward(node.grad)`
4. Each `GradFn` accumulates into its saved input tensors' `.grad` fields

**GradFn implementations per op:**

| Op | dInput computation |
|---|---|
| `MatMulBackward { a, b }` | `dA = dOut @ Bᵀ`, `dB = Aᵀ @ dOut` |
| `AddBackward` | `da = dOut`, `db = dOut` |
| `MulBackward { a, b }` | `da = dOut * b`, `db = dOut * a` |
| `TanhBackward { output }` | `dIn = dOut * (1 - output²)` |
| `SoftmaxBackward { output }` | Jacobian-vector product |
| `LayerNormBackward { x, mean, rstd, gamma }` | standard LN grad |
| `GeluBackward { x }` | `dIn = dOut * (Φ(x) + x·φ(x))` |
| `Conv2dBackward { input, weight }` | `dInput` = transposed conv, `dWeight` = cross-correlation |
| `BatchNormBackward { x, mean, var, gamma }` | standard BN grad |
| `ReLUBackward { mask }` | `dIn = dOut * mask` (mask=1 where input>0) |

---

## Phase 3 — `tensor` crate: CPU + GPU Ops

**Files:**
- `crates/tensor/src/ops/mod.rs`
- `crates/tensor/src/ops/cpu.rs`       — ndarray-backed ops
- `crates/tensor/src/ops/gpu.rs`       — wgpu compute dispatch
- `crates/tensor/shaders/matmul.wgsl`
- `crates/tensor/shaders/conv2d.wgsl`
- `crates/tensor/shaders/softmax.wgsl`
- `crates/tensor/shaders/elementwise.wgsl`

**`matmul.wgsl`** — tiled 16×16 workgroup matrix multiply:
```wgsl
@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
struct Uniforms { M: u32, N: u32, K: u32 }
@group(0) @binding(3) var<uniform> u: Uniforms;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) { ... }
```

**`conv2d.wgsl`** — im2col matmul or direct convolution kernel:
- Uniforms: N, C_in, C_out, H, W, kH, kW, pad, stride
- Each thread computes one output element

**`softmax.wgsl`** — numerically stable row-wise: subtract max, exp, divide by sum.

Each public op function (`matmul`, `conv2d`, `relu`, etc.) in `ops/mod.rs`:
1. Checks `tensor.device()` → dispatches to `cpu::*` or `gpu::*`
2. Creates output `Tensor` with appropriate `GradFn` attached

---

## Phase 4 — `tensor` crate: NN Layers

**Files:**
- `crates/tensor/src/nn/mod.rs`
- `crates/tensor/src/nn/linear.rs`
- `crates/tensor/src/nn/layer_norm.rs`
- `crates/tensor/src/nn/conv2d.rs`
- `crates/tensor/src/nn/batch_norm.rs`
- `crates/tensor/src/nn/attention.rs`
- `crates/tensor/src/nn/transformer.rs`

**`Conv2d`:** weight `[C_out, C_in, kH, kW]` + bias `[C_out]`, both `requires_grad=true`.

**`BatchNorm2d`:** running mean/var (not learned, updated during forward), learnable `gamma [C]`
and `beta [C]`. During training: normalize over N,H,W dims. During inference: use running stats.

**`Linear`:** weight `[out, in]` + bias `[out]`.

**`LayerNorm`:** normalize last dim. Learnable `gamma [d]`, `beta [d]`.

**`MultiHeadAttention`:** 4 Linear projections (Wq, Wk, Wv, Wo).
- Split into h=8 heads, each d_k=32 (8×32=256=d_model)
- `Attention(Q,K,V) = softmax(QKᵀ / √32) @ V`
- Concat heads → project via Wo

**`TransformerBlock`:** Attention→Add+Norm→FFN(GELU, 256→1024→256)→Add+Norm

**`TransformerEncoder`:** N stacked `TransformerBlock`s (independent weights per block)

All layers expose `fn parameters(&self) -> Vec<Tensor>` for the optimizer.

---

## Phase 5 — `tensor` crate: Adam Optimizer

**File:** `crates/tensor/src/optim/adam.rs`

```rust
pub struct Adam {
    params: Vec<Tensor>,  // all requires_grad=true tensors
    lr: f32,              // default 1e-4
    beta1: f32,           // 0.9
    beta2: f32,           // 0.999
    eps: f32,             // 1e-8
    m: Vec<Tensor>,       // first moment (same shape as params)
    v: Vec<Tensor>,       // second moment
    t: usize,             // step counter (for bias correction)
}
// step(): m = β1·m + (1-β1)·grad
//         v = β2·v + (1-β2)·grad²
//         m̂ = m/(1-β1^t),  v̂ = v/(1-β2^t)
//         param -= lr · m̂ / (√v̂ + ε)
// zero_grad(): set all .grad fields to None
```

---

## Phase 6 — `engine` crate: Board Encoding

**File:** `crates/engine/src/encode.rs`

Board → `[17 × 8 × 8]` float tensor:

```
Planes 0–5:   white { pawn, knight, bishop, rook, queen, king }  (binary)
Planes 6–11:  black { pawn, knight, bishop, rook, queen, king }  (binary)
Plane 12:     side_to_move  (1.0 everywhere if White, 0.0 if Black)
Plane 13:     castling WK   (1.0 everywhere if right available)
Plane 14:     castling WQ
Plane 15:     castling BK
Plane 16:     castling BQ
```

Uses `Board::piece_at(sq)` from `crates/chess/src/board.rs:piece_at`.
Metadata is baked directly into the input planes — no separate CLS injection needed.

---

## Phase 7 — `engine` crate: Hybrid Model

**Files:**
- `crates/engine/src/nn/resnet.rs`
- `crates/engine/src/model.rs`

**`resnet.rs`:**
```rust
pub struct ResidualBlock {
    conv1: Conv2d,   // (256, 256, 3×3, pad=1)
    bn1:   BatchNorm2d,
    conv2: Conv2d,   // (256, 256, 3×3, pad=1)
    bn2:   BatchNorm2d,
}
impl ResidualBlock {
    fn forward(&self, x: &Tensor) -> Tensor {
        let out = self.bn2.forward(&self.conv2.forward(
                    &relu(&self.bn1.forward(&self.conv1.forward(x)))));
        relu(&ops::add(&out, x))   // skip connection
    }
}
```

**`model.rs` — `HybridValueNet`:**
```rust
pub struct HybridValueNet {
    // CNN backbone
    conv_in:  Conv2d,        // (17→256, 3×3, pad=1)
    bn_in:    BatchNorm2d,
    blocks:   Vec<ResidualBlock>,  // ×8

    // Bridge
    cls_token: Tensor,       // [1×256] learnable
    pos_embed: Tensor,       // [65×256] learnable

    // Transformer head
    encoder:   TransformerEncoder,  // 4 blocks, d=256, h=8, ff=1024

    // Output
    head:      Linear,       // 256→1
}

impl HybridValueNet {
    pub fn forward(&self, board: &Board) -> Tensor {
        // 1. encode board → [17×8×8]
        let x = encode(board);
        // 2. CNN backbone
        let x = relu(&self.bn_in.forward(&self.conv_in.forward(&x)));  // [256×8×8]
        let x = self.blocks.iter().fold(x, |a, b| b.forward(&a));      // [256×8×8]
        // 3. reshape → [64×256]
        let x = x.reshape(&[64, 256]);
        // 4. prepend CLS → [65×256]
        let x = ops::cat(&[&self.cls_token, &x], 0);
        // 5. add positional embeddings
        let x = ops::add(&x, &self.pos_embed);
        // 6. transformer
        let x = self.encoder.forward(&x);                               // [65×256]
        // 7. CLS → scalar
        let cls = x.row(0);                                             // [256]
        ops::tanh(&self.head.forward(&cls))                             // scalar
    }
}
```

---

## Phase 8 — `engine` crate: PGN Parser & Dataset

**Files:**
- `crates/engine/src/pgn.rs`
- `crates/engine/src/dataset.rs`

**`pgn.rs`:** Minimal parser — no external dep.
- Extracts move list and result tag (`1-0` → 1.0, `0-1` → -1.0, `1/2-1/2` → 0.0)
- Replays moves via `chess::movegen::generate_legal_moves` + `Board::make_move`
- Emits `Vec<(Board, f32)>`

**`dataset.rs`:** `ChessDataset { samples: Vec<(Board, f32)> }`
- `fn shuffle(&mut self, seed: u64)`
- `fn batches(&self, size: usize) -> impl Iterator<Item=&[(Board, f32)]>`

---

## Phase 9 — `engine` crate: Training Loop

**File:** `crates/engine/src/train.rs`

```rust
pub struct TrainConfig {
    pub pgn_path:   PathBuf,
    pub epochs:     usize,      // default 20
    pub batch_size: usize,      // default 32
    pub lr:         f32,        // default 1e-4
    pub output:     PathBuf,
    pub device:     Device,
}

pub fn train(cfg: TrainConfig) {
    let mut model  = HybridValueNet::new(&cfg.device);
    let mut adam   = Adam::new(model.parameters(), cfg.lr);
    let mut dataset = ChessDataset::from_pgn(&cfg.pgn_path);

    for epoch in 0..cfg.epochs {
        dataset.shuffle(epoch as u64);
        let mut total_loss = 0f32;
        for batch in dataset.batches(cfg.batch_size) {
            adam.zero_grad();
            let loss = batch.iter()
                .map(|(board, label)| {
                    let pred = model.forward(board);
                    let diff = ops::sub(&pred, &Tensor::scalar(*label));
                    ops::mul(&diff, &diff)   // MSE per sample
                })
                .reduce(|a, b| ops::add(&a, &b))
                .unwrap();
            loss.backward();
            adam.step();
            total_loss += loss.item();
        }
        println!("Epoch {epoch}: loss = {:.5}", total_loss / dataset.len() as f32);
    }
    model.save(&cfg.output);
}
```

---

## Phase 10 — `engine` crate: Self-Play

**File:** `crates/engine/src/selfplay.rs`

Greedy self-play (no MCTS — can be added later):
1. `Board::starting_position()`
2. Each ply: `generate_legal_moves(&board)` → for each move, apply it, run
   `model.forward(&child_board)`, pick best value from side-to-move's perspective
3. Detect terminal (checkmate / stalemate / 50-move rule via `game_status`)
4. Label all positions in game with outcome
5. Return `Vec<(Board, f32)>` — feed into `ChessDataset` for training

---

## Phase 11 — CLI Subcommands

**File:** `crates/cli/src/main.rs` (modify existing `main.rs`)

```
cargo run -p cli -- train    --games games.pgn --epochs 20 --lr 1e-4 --device metal --output model.bin
cargo run -p cli -- eval     --model model.bin --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
cargo run -p cli -- selfplay --model model.bin --games 500 --output selfplay.pgn
```

`--device` accepts `cpu` or `metal` (maps to `Device::Cpu` / `Device::Gpu(GpuContext::new())`).

---

## Verification

| Test | How |
|---|---|
| Tensor matmul | Unit test: hand-verify 2×2 result |
| Autograd correctness | Numerical gradient check: compare `backward()` vs `(f(x+ε)-f(x-ε))/2ε` for all ops |
| GPU/CPU parity | Same matmul on both backends, assert within 1e-4 |
| Conv2d shape | Assert `conv2d([1,17,8,8], [256,17,3,3], pad=1)` → `[1,256,8,8]` |
| ResidualBlock | Assert output shape = input shape `[N,256,8,8]` |
| Encoding | `encode(Board::starting_position())` → shape `[17,8,8]`, planes 0-5 sum to 8 each |
| Forward pass | `model.forward(board)` → scalar in `(-1.0, 1.0)` |
| Training sanity | Loss strictly decreases over 5 epochs on 10-game dataset |
| Self-play | `selfplay --games 5` completes, produces valid FEN positions |
| CI | `cargo test --all` and `cargo lint` pass clean |

---

## Build Order

```
 1. tensor: Storage, Tensor, GpuContext (CPU + wgpu init)
 2. tensor: CPU ops — matmul, add, mul, relu, softmax (ndarray)
 3. tensor: Autograd — GradFn trait, topological backward, gradient check tests
 4. tensor: GPU ops — matmul.wgsl, elementwise.wgsl, softmax.wgsl (wgpu dispatch)
 5. tensor: Conv2d op — cpu::conv2d (im2col), conv2d.wgsl, Conv2dBackward GradFn
 6. tensor: BatchNorm op — cpu::batch_norm, BatchNormBackward GradFn
 7. tensor: NN layers — Linear, LayerNorm, Conv2d, BatchNorm2d
 8. tensor: NN layers — MultiHeadAttention, TransformerBlock, TransformerEncoder
 9. tensor: Adam optimizer
10. engine: encode.rs — Board → [17×8×8]
11. engine: nn/resnet.rs — ResidualBlock
12. engine: model.rs — HybridValueNet (full forward pass)
13. engine: pgn.rs + dataset.rs
14. engine: train.rs
15. engine: selfplay.rs
16. cli:    train / eval / selfplay subcommands
```
