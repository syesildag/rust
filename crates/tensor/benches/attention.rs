// Benchmarks for MultiHeadAttention forward paths.
//
// Compares the fused batched forward (2 GPU calls/block) against the per-head
// loop (16 GPU calls/block) so you can measure the regression if either path
// is accidentally reverted.
//
// Run:
//   cargo bench -p tensor --bench attention

#![allow(clippy::pedantic)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use tensor::nn::MultiHeadAttention;
use tensor::{ops, Tensor};

const BATCH: usize = 32;
const SEQ: usize = 65;
const D_MODEL: usize = 256;
const NUM_HEADS: usize = 8;

fn randn_grad(shape: &[usize]) -> Tensor {
    Tensor::randn(shape, 1.0).with_grad()
}

// ── forward_batched (fused) ───────────────────────────────────────────────────
//
// This is the optimised path: all heads computed in two batched matmuls.

fn bench_attention_batched(c: &mut Criterion) {
    let mut g = c.benchmark_group("attention");

    g.bench_function("forward_batched fused [B,S,D]", |b| {
        let attn = MultiHeadAttention::new(D_MODEL, NUM_HEADS);
        b.iter_batched(
            || randn_grad(&[BATCH, SEQ, D_MODEL]),
            |x| attn.forward_batched(&x, BATCH),
            BatchSize::SmallInput,
        );
    });

    g.bench_function("forward_batched fused forward+backward [B,S,D]", |b| {
        let attn = MultiHeadAttention::new(D_MODEL, NUM_HEADS);
        b.iter_batched(
            || randn_grad(&[BATCH, SEQ, D_MODEL]),
            |x| {
                let out = attn.forward_batched(&x, BATCH);
                let loss = ops::sum(&out);
                loss.backward();
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

// ── forward (single-sequence, per-head loop) ─────────────────────────────────
//
// The non-batched path is untouched by the optimisation.
// Benchmarking it gives a baseline and catches accidental regressions.

fn bench_attention_single(c: &mut Criterion) {
    let mut g = c.benchmark_group("attention_single");

    g.bench_function("forward per-head [S,D]", |b| {
        let attn = MultiHeadAttention::new(D_MODEL, NUM_HEADS);
        b.iter_batched(
            || randn_grad(&[SEQ, D_MODEL]),
            |x| attn.forward(&x),
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

criterion_group!(benches, bench_attention_batched, bench_attention_single);
criterion_main!(benches);
