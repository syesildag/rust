# Training Performance Optimization — Design Spec

**Date:** 2026-05-09  
**Status:** Approved

## Problem

Profiler data shows the training loop spending 59% of wall time in `Tensor::backward`, with `MatMulBatchedBackward` (13%) and `Conv2dBackward` (13%) as the dominant ops. Three root causes are identified:

1. **Systemic grad-clone overhead** — every backward node allocates a full Vec copy of its gradient before dispatching.
2. **Missing saved-data patterns** — `BatchNorm2dBackward` and `Conv2dBackward` re-read tensor data through RwLocks during backward instead of saving it at forward time; `BatchNorm2dBackward` also allocates per-channel temp Vecs in a sequential loop.
3. **GPU bypass in backward** — `MatMulBatchedBackward` calls `cpu_matmul_2d` directly, skipping the GPU dispatch path that the forward already uses.
4. **Serialized per-head attention** — `MultiHeadAttention::forward_batched` issues 16 GPU command submissions per transformer block (8 per-head × 2 matmuls), where a single fused `[B*H, S, d_k]` batched matmul would issue 2.

## Scope

Changes are confined to:

```
tensor/src/tensor_impl.rs   — grad clone elimination
tensor/src/ops.rs           — BatchNorm/Conv2d/dropout fixes, MatMulBatched
                              backward GPU dispatch, new permute_4d op
tensor/src/nn/attention.rs  — fused multi-head attention
```

No changes to `engine/`, `chess/`, `cli/`, or any public API. One new public function `ops::permute_4d` is added.

## Design

### 1. Grad clone elimination (`tensor_impl.rs`)

**Current:**
```rust
let grad = t.inner.grad.lock().expect("...").clone();
gfn.backward(&grad);
```

**Fix:** Hold the MutexGuard and pass `&*guard` as `&[f32]` directly. Safe because `gfn.backward()` only accumulates into *input* tensors — topological ordering guarantees `t` is never its own input, so no deadlock.

```rust
let guard = t.inner.grad.lock().expect("...");
gfn.backward(&guard);
```

### 2. CPU overhead fixes (`ops.rs`)

#### 2a. `BatchNorm2dBackward`

- **Add `gamma_data: Vec<f32>`** to the struct, populated at forward time. Eliminates `self.gamma.data()` (RwLock + clone) on every backward call.
- **Eliminate per-channel temp Vecs** (`x_hat_c`, `g_c`): replaced by direct strided indexing into `saved_x_hat` and `g` using `idx = ni * c * h * w + ci * h * w + hi * w + wi`.
- **Parallelize channel loop** with `(0..c).into_par_iter()`: each channel's contribution to `d_input`, `d_gamma`, `d_beta` is independent. Partial `d_input` slices are computed per channel and merged after the parallel section.

#### 2b. `BatchNorm2d` forward

- **Parallelize the channel loop** with `(0..c).into_par_iter()` in the forward pass as well. Each channel computes mean, variance, `inv_std`, and writes its slice of `data`/`saved_x_hat` independently.

#### 2c. `Conv2dBackward`

- **Add `weight_data: Vec<f32>`** to the struct, populated at forward time. Eliminates `self.weight.data()` (RwLock + clone) on every backward call. Follows the identical pattern already used in `LinearBackward`.

#### 2d. `dropout`

- **Single `x.data()` read** — store the result, use it for both mask generation and output computation.

### 3. Backward GPU dispatch (`MatMulBatchedBackward`, `ops.rs`)

**Current:** Per-batch-slice loop calling `cpu_matmul_2d` (CPU-only).

**Fix:** Restructure both gradient matmuls to operate on the full `[batch*m, …]` flat layout and route through `matmul_2d` (GPU-aware dispatch). At the model's training dimensions (batch=32, seq=65, d_k=32), each backward matmul is ~4.3M FLOPs — above the 1M GPU threshold.

- `da`: `matmul_2d(g_reshaped, b_transposed, batch*m, n, k)` — shape `[batch*m, k]`
- `db`: `matmul_2d(a_transposed, g_reshaped, k, batch*m, n)` — shape `[k, batch*n]`, then transposed

This mirrors the structure already used in `LinearBackward`.

### 4. `ops::permute_4d` (new op, `ops.rs`)

```rust
pub fn permute_4d(x: &Tensor, axes: [usize; 4]) -> Tensor
```

Materialises a permuted copy of a 4-D tensor. The backward applies the inverse permutation to `grad_output`. Panics if `x` is not 4-D or `axes` is not a permutation of `[0,1,2,3]`.

**`Permute4dBackward` struct:**
```rust
struct Permute4dBackward { input: Tensor, axes: [usize; 4], shape: Vec<usize> }
```
Backward: compute `inv_axes` from `axes`, call `permute_4d_data(grad_output, inv_axes, grad_shape)`.

### 5. Fused multi-head attention (`attention.rs`)

**Target:** `MultiHeadAttention::forward_batched` only. The single-sequence `forward` path is unchanged.

**Current:** `num_heads` iterations, each issuing 2 `matmul_batched` GPU calls → 16 GPU submissions per block.

**New flow:**

```
q_all [B*S, D]
  → reshape [B, S, H, d_k]
  → permute_4d([0,2,1,3]) → [B, H, S, d_k]
  → reshape [B*H, S, d_k]       ← q_heads

k_heads, v_heads: same reshape chain

scores = matmul_batched(q_heads, transpose_last_two(k_heads)) * scale
       → [B*H, S, S]
scores = clamp(scores, -30, 30)
attn   = softmax(scores reshape [B*H*S, S]) reshape [B*H, S, S]
ctx    = matmul_batched(attn, v_heads)      → [B*H, S, d_k]

ctx reshape [B, H, S, d_k]
  → permute_4d([0,2,1,3]) → [B, S, H, d_k]
  → reshape [B*S, D]
  → wo.forward(...)
  → reshape [B, S, D]
```

GPU submissions per block: **2** (down from 16). Across 4 blocks: **8** (down from 64).

## Testing

| Test | Location | What it checks |
|------|----------|----------------|
| `permute_4d_roundtrip` | `ops.rs` | shape and round-trip identity |
| `fused_attention_matches_per_head` | `attention.rs` | fused output ≡ per-head output within f32 tolerance |
| All existing op tests | `ops.rs`, `tensor_impl.rs` | no regression in backward correctness |
| `cargo lint` | CI | no new Clippy warnings |

## Expected Impact

| Change | Profiler share affected | Expected reduction |
|--------|------------------------|--------------------|
| Grad clone elimination | All backward (59%) | 5–10% of total |
| BatchNorm CPU fixes | 8.7% | ~4% of total |
| Conv2d saved weight | 13% | ~2% of total |
| Dropout single read | <1% | negligible |
| MatMulBatched backward GPU | 13% | 8–12% of total |
| Fused attention | MatMulBatched forward + overhead | 10–15% of total |
| **Total estimated** | | **~30–45% wall-clock reduction** |
