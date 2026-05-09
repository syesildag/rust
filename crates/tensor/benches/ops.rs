// Benchmarks for the ops changed in the training-performance optimisation.
//
// Shapes match the model's real training dimensions so the GPU-dispatch
// threshold (1M FLOPs) is crossed and GPU paths are exercised.
//
// Run:
//   cargo bench -p tensor
//   cargo bench -p tensor -- matmul_batched   # single group
//   cargo bench -p tensor -- --save-baseline main   # save a baseline
//   cargo bench -p tensor -- --baseline main        # compare against it

#![allow(clippy::pedantic)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use tensor::{ops, Tensor};

// ── Model constants ──────────────────────────────────────────────────────────

const BATCH: usize = 32;
const SEQ: usize = 65; // 64 squares + CLS
const D_MODEL: usize = 256;
const NUM_HEADS: usize = 8;
const D_K: usize = D_MODEL / NUM_HEADS; // 32
const CHANNELS: usize = 256;
const IMG: usize = 8;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn randn(shape: &[usize]) -> Tensor {
    Tensor::randn(shape, 1.0)
}

fn randn_grad(shape: &[usize]) -> Tensor {
    Tensor::randn(shape, 1.0).with_grad()
}

// ── matmul_batched ───────────────────────────────────────────────────────────
//
// Forward: [B*H, S, d_k] × [B*H, d_k, S] → [B*H, S, S]
// FLOPs ≈ 2 × B*H × S × d_k × S = 2 × 256 × 65 × 32 × 65 ≈ 69 M
// Well above the 1 M GPU threshold, so GPU path is exercised.

fn bench_matmul_batched(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_batched");

    g.bench_function("forward [B*H,S,d_k] x [B*H,d_k,S]", |b| {
        b.iter_batched(
            || {
                let q = randn(&[BATCH * NUM_HEADS, SEQ, D_K]);
                let k = randn(&[BATCH * NUM_HEADS, D_K, SEQ]);
                (q, k)
            },
            |(q, k)| ops::matmul_batched(&q, &k),
            BatchSize::SmallInput,
        );
    });

    // Backward pass: requires grad-enabled inputs so autograd builds the graph.
    g.bench_function("forward+backward [B*H,S,d_k] x [B*H,d_k,S]", |b| {
        b.iter_batched(
            || {
                let q = randn_grad(&[BATCH * NUM_HEADS, SEQ, D_K]);
                let k = randn_grad(&[BATCH * NUM_HEADS, D_K, SEQ]);
                (q, k)
            },
            |(q, k)| {
                let out = ops::matmul_batched(&q, &k);
                let loss = ops::sum(&out);
                loss.backward();
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

// ── batch_norm_2d ─────────────────────────────────────────────────────────────
//
// Exercises the parallelised channel loop added in the optimisation.
// Shape [B, C, H, W] = [32, 256, 8, 8].

fn bench_batch_norm_2d(c: &mut Criterion) {
    let mut g = c.benchmark_group("batch_norm_2d");

    g.bench_function("forward [B,C,H,W]", |b| {
        b.iter_batched(
            || {
                let x = randn(&[BATCH, CHANNELS, IMG, IMG]);
                let gamma = Tensor::ones(&[CHANNELS]);
                let beta = Tensor::zeros(&[CHANNELS]);
                (x, gamma, beta)
            },
            |(x, gamma, beta)| ops::batch_norm_2d(&x, &gamma, &beta, 1e-5),
            BatchSize::SmallInput,
        );
    });

    g.bench_function("forward+backward [B,C,H,W]", |b| {
        b.iter_batched(
            || {
                let x = randn_grad(&[BATCH, CHANNELS, IMG, IMG]);
                let gamma = Tensor::ones(&[CHANNELS]).with_grad();
                let beta = Tensor::zeros(&[CHANNELS]).with_grad();
                (x, gamma, beta)
            },
            |(x, gamma, beta)| {
                let out = ops::batch_norm_2d(&x, &gamma, &beta, 1e-5);
                let loss = ops::sum(&out);
                loss.backward();
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

// ── permute_4d ───────────────────────────────────────────────────────────────
//
// The [0,2,1,3] permutation used in the fused attention path.
// Shape [B, H, S, d_k] = [32, 8, 65, 32].

fn bench_permute_4d(c: &mut Criterion) {
    let mut g = c.benchmark_group("permute_4d");

    g.bench_function("forward [B,H,S,d_k] axes=[0,2,1,3]", |b| {
        b.iter_batched(
            || randn(&[BATCH, NUM_HEADS, SEQ, D_K]),
            |x| ops::permute_4d(&x, [0, 2, 1, 3]),
            BatchSize::SmallInput,
        );
    });

    g.bench_function("forward+backward [B,H,S,d_k] axes=[0,2,1,3]", |b| {
        b.iter_batched(
            || randn_grad(&[BATCH, NUM_HEADS, SEQ, D_K]),
            |x| {
                let out = ops::permute_4d(&x, [0, 2, 1, 3]);
                let loss = ops::sum(&out);
                loss.backward();
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

// ── conv2d ────────────────────────────────────────────────────────────────────
//
// Backward uses the saved weight_data optimisation.
// Shape: [B, C_in=18, 8, 8] with 256 output channels, 3×3 kernel.

fn bench_conv2d(c: &mut Criterion) {
    let mut g = c.benchmark_group("conv2d");

    const C_IN: usize = 18;
    const C_OUT: usize = 256;
    const K: usize = 3;

    g.bench_function("forward+backward [B,18,8,8]", |b| {
        b.iter_batched(
            || {
                let x = randn_grad(&[BATCH, C_IN, IMG, IMG]);
                let w = randn_grad(&[C_OUT, C_IN, K, K]);
                let bias = Tensor::zeros(&[C_OUT]).with_grad();
                (x, w, bias)
            },
            |(x, w, bias)| {
                let out = ops::conv2d(&x, &w, &bias, 1);
                let loss = ops::sum(&out);
                loss.backward();
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

// ── backward pass (grad-clone elimination) ────────────────────────────────────
//
// A small linear chain: matmul → relu → matmul → sum → backward.
// The grad-clone elimination applies on every backward node traversal, so
// deeper graphs benefit more.

fn bench_backward(c: &mut Criterion) {
    let mut g = c.benchmark_group("backward");

    g.bench_function("linear_chain depth=4", |b| {
        b.iter_batched(
            || {
                let x = randn_grad(&[BATCH * SEQ, D_MODEL]);
                let w1 = randn_grad(&[D_MODEL, D_MODEL]);
                let w2 = randn_grad(&[D_MODEL, D_MODEL]);
                (x, w1, w2)
            },
            |(x, w1, w2)| {
                let h = ops::relu(&ops::matmul(&x, &w1));
                let out = ops::matmul(&h, &w2);
                let loss = ops::sum(&out);
                loss.backward();
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_matmul_batched,
    bench_batch_norm_2d,
    bench_permute_4d,
    bench_conv2d,
    bench_backward,
);
criterion_main!(benches);
