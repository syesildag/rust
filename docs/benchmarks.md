# Benchmarks

The `tensor` crate includes Criterion benchmarks covering the five operations
changed in the training-performance optimisation. They live in
`crates/tensor/benches/` and run as part of `cargo bench`.

---

## Running the benchmarks

```bash
# Run all benchmarks (no comparison)
cargo bench -p tensor

# Run a single bench file
cargo bench -p tensor --bench ops
cargo bench -p tensor --bench attention

# Filter to a specific group within a file
cargo bench -p tensor --bench ops -- matmul_batched

# Save a named baseline (run once after a known-good state)
cargo bench -p tensor --bench ops -- --save-baseline main
cargo bench -p tensor --bench attention -- --save-baseline main

# Compare every subsequent run against that baseline
cargo bench -p tensor --bench ops -- --baseline main
cargo bench -p tensor --bench attention -- --baseline main
```

Criterion prints a confidence interval for each benchmark and highlights
regressions when a `--baseline` is active. HTML reports are written to
`target/criterion/` after every run.

---

## Bench files

### `benches/ops.rs`

Covers the five ops modified in the training-performance work:

| Group | Shape | What it exercises |
|---|---|---|
| `matmul_batched` | `[B*H, S, d_k] × [B*H, d_k, S]` | GPU-dispatched batched matmul forward + backward |
| `batch_norm_2d` | `[B, C, H, W]` | Parallel channel loop (rayon) forward + backward |
| `permute_4d` | `[B, H, S, d_k]` axes `[0,2,1,3]` | New op used by fused attention, forward + backward |
| `conv2d` | `[B, 18, 8, 8]` 3×3 → 256ch | Saved `weight_data` in backward |
| `backward` | depth-4 linear chain | Grad-clone elimination across the backward pass |

Each group benchmarks forward-only and forward+backward separately so you
can distinguish where time is spent.

### `benches/attention.rs`

| Group | Shape | What it exercises |
|---|---|---|
| `attention` | `[B, S, D]` | Fused `forward_batched`: 2 GPU calls per block |
| `attention_single` | `[S, D]` | Single-sequence `forward`: per-head loop (unchanged) |

---

## Shapes and why they matter

All shapes match the model's actual training dimensions:

| Symbol | Value | Meaning |
|---|---|---|
| B | 32 | batch size |
| S | 65 | sequence length (64 squares + CLS token) |
| D | 256 | model dimension |
| H | 8 | attention heads |
| d_k | 32 | per-head dimension (D / H) |
| C | 256 | ResNet channel count |
| img | 8×8 | board spatial size |

Benchmarking at real training shapes matters because the GPU dispatch path
only activates above 1 M FLOPs. At toy sizes the CPU fallback runs instead,
which would measure the wrong thing.

---

## Baseline numbers (2026-05-09, Apple M-series CPU, Metal backend)

Recorded immediately after the training-performance optimisation landed on `main`.

| Benchmark | Median |
|---|---|
| `matmul_batched/forward` | ~10 ms |
| `matmul_batched/forward+backward` | ~24 ms |
| `batch_norm_2d/forward` | ~1.0 ms |
| `batch_norm_2d/forward+backward` | ~2.3 ms |
| `permute_4d/forward` | ~0.96 ms |
| `permute_4d/forward+backward` | ~2.3 ms |
| `conv2d/forward+backward` | ~17 ms |
| `backward/linear_chain depth=4` | see Criterion output |
| `attention/fused fwd+bwd` | ~179 ms |
| `attention_single/forward per-head` | ~7.6 ms |

These numbers are wall-clock including CPU↔GPU command encoding. They are not
pure GPU compute times. Use them as regression guards, not absolute performance
figures.

---

## Saving a permanent baseline

After a significant change (optimisation, regression fix), save a named
baseline so future runs can compare against it:

```bash
cargo bench -p tensor --bench ops -- --save-baseline <tag>
cargo bench -p tensor --bench attention -- --save-baseline <tag>
```

Baselines are stored in `target/criterion/` and are not tracked in git
(covered by `.gitignore`). To share them, export the HTML reports instead.
