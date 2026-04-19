//! Transformer encoder block and stacked encoder.

use crate::nn::{LayerNorm, Linear, MultiHeadAttention};
use crate::ops;
use crate::Tensor;

/// A single transformer encoder block.
///
/// Architecture:
/// ```text
/// x → MHA(x) → Add+Norm → FFN → Add+Norm
/// ```
pub struct TransformerBlock {
    attn: MultiHeadAttention,
    norm1: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    norm2: LayerNorm,
}

impl TransformerBlock {
    /// Creates a new transformer block.
    ///
    /// `d_ff` is the inner FFN dimension, typically 4× `d_model`.
    #[must_use]
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            attn: MultiHeadAttention::new(d_model, num_heads),
            norm1: LayerNorm::new(d_model),
            ff1: Linear::new(d_model, d_ff, true),
            ff2: Linear::new(d_ff, d_model, true),
            norm2: LayerNorm::new(d_model),
        }
    }

    /// Applies self-attention + FFN with residual connections.
    ///
    /// Input/output shape: `[seq_len, d_model]`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Attention sub-layer with residual.
        let attn_out = self.attn.forward(x);
        let x = self.norm1.forward(&ops::add(x, &attn_out));
        // Feed-forward sub-layer with residual.
        let ff_out = self.ff2.forward(&ops::gelu(&self.ff1.forward(&x)));
        self.norm2.forward(&ops::add(&x, &ff_out))
    }

    /// Forward for a batch of sequences: `[B, seq, d_model] → [B, seq, d_model]`.
    #[must_use]
    pub fn forward_batched(&self, x: &Tensor, batch: usize) -> Tensor {
        let shape = x.shape();
        let (seq, d_model) = (shape[1], shape[2]);

        // Attention sub-layer with residual + layer norm.
        let attn_out = self.attn.forward_batched(x, batch);
        let x_res = ops::add(x, &attn_out);
        let x = self
            .norm1
            .forward(&x_res.reshape(&[batch * seq, d_model]))
            .reshape(&[batch, seq, d_model]);

        // FFN sub-layer with residual + layer norm.
        let x_2d = x.reshape(&[batch * seq, d_model]);
        let ff_out = self
            .ff2
            .forward(&ops::gelu(&self.ff1.forward(&x_2d)))
            .reshape(&[batch, seq, d_model]);
        let x_res2 = ops::add(&x, &ff_out);
        self.norm2
            .forward(&x_res2.reshape(&[batch * seq, d_model]))
            .reshape(&[batch, seq, d_model])
    }

    /// Returns all learnable parameters from this block.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.attn.parameters();
        p.extend(self.norm1.parameters());
        p.extend(self.ff1.parameters());
        p.extend(self.ff2.parameters());
        p.extend(self.norm2.parameters());
        p
    }
}

/// Stacked transformer encoder: `N` independent `TransformerBlock`s.
pub struct TransformerEncoder {
    blocks: Vec<TransformerBlock>,
}

impl TransformerEncoder {
    /// Creates a stack of `num_layers` transformer blocks.
    #[must_use]
    pub fn new(num_layers: usize, d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        let blocks = (0..num_layers)
            .map(|_| TransformerBlock::new(d_model, num_heads, d_ff))
            .collect();
        Self { blocks }
    }

    /// Applies all transformer blocks in sequence.
    ///
    /// Input/output shape: `[seq_len, d_model]`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.blocks
            .iter()
            .fold(x.clone(), |acc, block| block.forward(&acc))
    }

    /// Applies all transformer blocks in sequence, batched.
    ///
    /// Input/output shape: `[B, seq_len, d_model]`.
    #[must_use]
    pub fn forward_batched(&self, x: &Tensor, batch: usize) -> Tensor {
        self.blocks
            .iter()
            .fold(x.clone(), |acc, block| block.forward_batched(&acc, batch))
    }

    /// Returns all learnable parameters from all blocks.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        self.blocks
            .iter()
            .flat_map(TransformerBlock::parameters)
            .collect()
    }
}
