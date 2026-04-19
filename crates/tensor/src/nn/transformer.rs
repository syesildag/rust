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
    /// Creates a transformer block.
    ///
    /// `d_ff` is the inner dimension of the feed-forward network (typically 4×`d_model`).
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

    /// Forward: `[S, d_model] → [S, d_model]`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Attention sub-layer with residual.
        let attn_out = self.attn.forward(x);
        let x = self.norm1.forward(&ops::add(x, &attn_out));
        // Feed-forward sub-layer with residual.
        let ff_out = self.ff2.forward(&ops::gelu(&self.ff1.forward(&x)));
        self.norm2.forward(&ops::add(&x, &ff_out))
    }

    /// Returns all learnable parameters in this block.
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
    /// Creates a stacked encoder with `num_layers` blocks.
    #[must_use]
    pub fn new(num_layers: usize, d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        let blocks = (0..num_layers)
            .map(|_| TransformerBlock::new(d_model, num_heads, d_ff))
            .collect();
        Self { blocks }
    }

    /// Forward: `[S, d_model] → [S, d_model]`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.blocks
            .iter()
            .fold(x.clone(), |acc, block| block.forward(&acc))
    }

    /// Returns all learnable parameters across all blocks.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        self.blocks
            .iter()
            .flat_map(TransformerBlock::parameters)
            .collect()
    }
}
