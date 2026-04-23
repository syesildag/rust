//! Transformer encoder block and stacked encoder.

use crate::nn::{Dropout, LayerNorm, Linear, MultiHeadAttention};
use crate::ops;
use crate::Tensor;

/// A single transformer encoder block.
///
/// Architecture:
/// ```text
/// x → MHA(x) → Dropout → Add+Norm → FFN → Dropout → Add+Norm
/// ```
pub struct TransformerBlock {
    attn: MultiHeadAttention,
    dropout1: Dropout,
    norm1: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    dropout2: Dropout,
    norm2: LayerNorm,
}

impl TransformerBlock {
    /// Creates a new transformer block.
    ///
    /// - `d_ff` is the inner FFN dimension, typically 4× `d_model`.
    /// - `dropout` is the drop probability applied after attention and after the FFN.
    #[must_use]
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, dropout: f32) -> Self {
        Self {
            attn: MultiHeadAttention::new(d_model, num_heads),
            dropout1: Dropout::new(dropout),
            norm1: LayerNorm::new(d_model),
            ff1: Linear::new(d_model, d_ff, true),
            ff2: Linear::new(d_ff, d_model, true),
            dropout2: Dropout::new(dropout),
            norm2: LayerNorm::new(d_model),
        }
    }

    /// Switches training/inference mode for the dropout layers in this block.
    pub fn set_training(&self, v: bool) {
        self.dropout1.set_training(v);
        self.dropout2.set_training(v);
    }

    /// Applies self-attention + FFN with residual connections and dropout.
    ///
    /// Input/output shape: `[seq_len, d_model]`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Attention sub-layer: dropout → residual → norm.
        let attn_out = self.dropout1.forward(&self.attn.forward(x));
        let x = self.norm1.forward(&ops::add(x, &attn_out));
        // Feed-forward sub-layer: dropout → residual → norm.
        let ff_out = self.dropout2.forward(&self.ff2.forward(&ops::gelu(&self.ff1.forward(&x))));
        self.norm2.forward(&ops::add(&x, &ff_out))
    }

    /// Forward for a batch of sequences: `[B, seq, d_model] → [B, seq, d_model]`.
    #[must_use]
    pub fn forward_batched(&self, x: &Tensor, batch: usize) -> Tensor {
        let shape = x.shape();
        let (seq, d_model) = (shape[1], shape[2]);

        // Attention sub-layer: dropout → residual → layer norm.
        let attn_out = self.dropout1.forward(&self.attn.forward_batched(x, batch));
        let x_res = ops::add(x, &attn_out);
        let x = self
            .norm1
            .forward(&x_res.reshape(&[batch * seq, d_model]))
            .reshape(&[batch, seq, d_model]);

        // FFN sub-layer: dropout → residual → layer norm.
        let x_2d = x.reshape(&[batch * seq, d_model]);
        let ff_out = self
            .dropout2
            .forward(
                &self
                    .ff2
                    .forward(&ops::gelu(&self.ff1.forward(&x_2d)))
                    .reshape(&[batch, seq, d_model]),
            );
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
    ///
    /// `dropout` is passed through to each [`TransformerBlock`].
    #[must_use]
    pub fn new(num_layers: usize, d_model: usize, num_heads: usize, d_ff: usize, dropout: f32) -> Self {
        let blocks = (0..num_layers)
            .map(|_| TransformerBlock::new(d_model, num_heads, d_ff, dropout))
            .collect();
        Self { blocks }
    }

    /// Switches training/inference mode for all blocks.
    pub fn set_training(&self, v: bool) {
        for block in &self.blocks {
            block.set_training(v);
        }
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
