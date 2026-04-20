//! Multi-head self-attention with fused Q/K/V projections.
//!
//! Instead of `h` separate `[d_model, d_k]` weight matrices per projection,
//! a single `[d_model, d_model]` matrix projects all heads at once. This cuts
//! GPU round-trips for projections from `3 * h` to `3` per forward pass.

#![allow(clippy::cast_precision_loss)]

use crate::nn::Linear;
use crate::ops;
use crate::Tensor;

/// Multi-head self-attention with fused Q/K/V projections.
///
/// `d_model` must be divisible by `num_heads`.
pub struct MultiHeadAttention {
    /// Fused query projection `[d_model, d_model]` — output split into `h` heads.
    wq: Linear,
    /// Fused key projection `[d_model, d_model]`.
    wk: Linear,
    /// Fused value projection `[d_model, d_model]`.
    wv: Linear,
    /// Output projection `[d_model, d_model]`.
    wo: Linear,
    /// Number of parallel attention heads.
    pub num_heads: usize,
    /// Dimension of each head's key/query/value projections (`d_model / num_heads`).
    pub d_k: usize,
}

impl MultiHeadAttention {
    /// Creates a new `MultiHeadAttention`.
    ///
    /// # Panics
    /// Panics if `d_model` is not divisible by `num_heads`.
    #[must_use]
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert_eq!(
            d_model % num_heads,
            0,
            "d_model must be divisible by num_heads"
        );
        let d_k = d_model / num_heads;
        Self {
            wq: Linear::new(d_model, d_model, false),
            wk: Linear::new(d_model, d_model, false),
            wv: Linear::new(d_model, d_model, false),
            wo: Linear::new(d_model, d_model, false),
            num_heads,
            d_k,
        }
    }

    /// Forward: `[seq, d_model] → [seq, d_model]`.
    ///
    /// One GPU call per Q/K/V projection (was `num_heads` calls each).
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let scale = 1.0 / (self.d_k as f32).sqrt();
        // Fused projections — 3 GPU calls regardless of num_heads.
        let q_all = self.wq.forward(x); // [seq, d_model]
        let k_all = self.wk.forward(x);
        let v_all = self.wv.forward(x);

        let head_tensors: Vec<Tensor> = (0..self.num_heads)
            .map(|head| {
                let (start, end) = (head * self.d_k, (head + 1) * self.d_k);
                let query = ops::slice_cols(&q_all, start, end); // [seq, d_k]
                let key = ops::slice_cols(&k_all, start, end);
                let val = ops::slice_cols(&v_all, start, end);
                let scores = ops::mul_scalar(&ops::matmul(&query, &key.t()), scale);
                let attn = ops::softmax(&scores);
                ops::matmul(&attn, &val) // [seq, d_k]
            })
            .collect();

        let refs: Vec<&Tensor> = head_tensors.iter().collect();
        self.wo.forward(&ops::cat_cols(&refs)) // [seq, d_model]
    }

    /// Forward for a batch of sequences: `[B, seq, d_model] → [B, seq, d_model]`.
    ///
    /// Projects Q/K/V for all `B*seq` tokens in one matmul each, then uses
    /// batched matmul for attention scores and context aggregation.
    #[must_use]
    pub fn forward_batched(&self, x: &Tensor, batch: usize) -> Tensor {
        let shape = x.shape();
        let (seq, d_model) = (shape[1], shape[2]);
        let scale = 1.0 / (self.d_k as f32).sqrt();
        let x_2d = x.reshape(&[batch * seq, d_model]);

        // Fused projections — 3 GPU calls regardless of num_heads.
        let q_all = self.wq.forward(&x_2d); // [B*seq, d_model]
        let k_all = self.wk.forward(&x_2d);
        let v_all = self.wv.forward(&x_2d);

        let head_tensors: Vec<Tensor> = (0..self.num_heads)
            .map(|head| {
                let (start, end) = (head * self.d_k, (head + 1) * self.d_k);
                let query = ops::slice_cols(&q_all, start, end).reshape(&[batch, seq, self.d_k]);
                let key = ops::slice_cols(&k_all, start, end).reshape(&[batch, seq, self.d_k]);
                let val = ops::slice_cols(&v_all, start, end).reshape(&[batch, seq, self.d_k]);
                let kt = ops::transpose_last_two(&key); // [B, d_k, seq]
                let scores = ops::mul_scalar(&ops::matmul_batched(&query, &kt), scale); // [B, seq, seq]
                let attn =
                    ops::softmax(&scores.reshape(&[batch * seq, seq])).reshape(&[batch, seq, seq]);
                ops::matmul_batched(&attn, &val).reshape(&[batch * seq, self.d_k])
                // [B*seq, d_k]
            })
            .collect();

        let refs: Vec<&Tensor> = head_tensors.iter().collect();
        let concat = ops::cat_cols(&refs); // [B*seq, d_model]
        self.wo.forward(&concat).reshape(&[batch, seq, d_model])
    }

    /// Returns all learnable parameters.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.wq.parameters();
        p.extend(self.wk.parameters());
        p.extend(self.wv.parameters());
        p.extend(self.wo.parameters());
        p
    }
}
