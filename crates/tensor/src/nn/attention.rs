//! Multi-head self-attention.
//!
//! Runs `h` attention heads in parallel (each of dimension `d_k = d_model / h`),
//! concatenates them, and applies a final output projection.

#![allow(clippy::cast_precision_loss)]

use crate::nn::Linear;
use crate::ops;
use crate::Tensor;

/// Multi-head self-attention.
///
/// `d_model` must be divisible by `num_heads`.
pub struct MultiHeadAttention {
    wq: Vec<Linear>,
    wk: Vec<Linear>,
    wv: Vec<Linear>,
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
        let wq = (0..num_heads)
            .map(|_| Linear::new(d_model, d_k, false))
            .collect();
        let wk = (0..num_heads)
            .map(|_| Linear::new(d_model, d_k, false))
            .collect();
        let wv = (0..num_heads)
            .map(|_| Linear::new(d_model, d_k, false))
            .collect();
        let wo = Linear::new(d_model, d_model, false);
        Self {
            wq,
            wk,
            wv,
            wo,
            num_heads,
            d_k,
        }
    }

    /// Forward: `[S, d_model] → [S, d_model]`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let scale = 1.0 / (self.d_k as f32).sqrt();
        // Compute each head: [S, d_k]
        let head_tensors: Vec<Tensor> = (0..self.num_heads)
            .map(|h| {
                let q = self.wq[h].forward(x);
                let k = self.wk[h].forward(x);
                let v = self.wv[h].forward(x);
                let scores = ops::mul_scalar(&ops::matmul(&q, &k.t()), scale);
                let attn = ops::softmax(&scores);
                ops::matmul(&attn, &v) // [S, d_k]
            })
            .collect();
        // Concatenate all heads along dim-1 → [S, d_model], then project.
        let refs: Vec<&Tensor> = head_tensors.iter().collect();
        let concat = ops::cat_cols(&refs); // [S, d_model]
        self.wo.forward(&concat)
    }

    /// Forward for a batch of sequences: `[B, seq, d_model] → [B, seq, d_model]`.
    ///
    /// Projects Q/K/V for all B*seq tokens in one matmul per head, then uses
    /// batched matmul for attention scores and context aggregation.
    #[must_use]
    pub fn forward_batched(&self, x: &Tensor, batch: usize) -> Tensor {
        let shape = x.shape();
        let (seq, d_model) = (shape[1], shape[2]);
        let scale = 1.0 / (self.d_k as f32).sqrt();
        let x_2d = x.reshape(&[batch * seq, d_model]);

        let head_tensors: Vec<Tensor> = (0..self.num_heads)
            .map(|head| {
                let query = self.wq[head].forward(&x_2d).reshape(&[batch, seq, self.d_k]);
                let key = self.wk[head].forward(&x_2d).reshape(&[batch, seq, self.d_k]);
                let val = self.wv[head].forward(&x_2d).reshape(&[batch, seq, self.d_k]);
                let kt = ops::transpose_last_two(&key); // [B, d_k, seq]
                let scores = ops::mul_scalar(&ops::matmul_batched(&query, &kt), scale); // [B, seq, seq]
                let attn = ops::softmax(&scores.reshape(&[batch * seq, seq]))
                    .reshape(&[batch, seq, seq]);
                ops::matmul_batched(&attn, &val).reshape(&[batch * seq, self.d_k]) // [B*seq, d_k]
            })
            .collect();

        let refs: Vec<&Tensor> = head_tensors.iter().collect();
        let concat = ops::cat_cols(&refs); // [B*seq, d_model]
        self.wo.forward(&concat).reshape(&[batch, seq, d_model])
    }

    /// Returns all learnable parameters.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for h in 0..self.num_heads {
            p.extend(self.wq[h].parameters());
            p.extend(self.wk[h].parameters());
            p.extend(self.wv[h].parameters());
        }
        p.extend(self.wo.parameters());
        p
    }
}
