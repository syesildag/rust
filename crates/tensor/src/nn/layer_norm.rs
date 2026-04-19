//! Layer normalisation over the last dimension.

use crate::ops;
use crate::Tensor;

/// Normalises `[S, D]` inputs along the `D` dimension.
/// Learnable `gamma` (scale) and `beta` (shift), both `[D]`.
pub struct LayerNorm {
    /// Learnable per-feature scale parameter, shape `[D]`.
    pub gamma: Tensor,
    /// Learnable per-feature shift parameter, shape `[D]`.
    pub beta: Tensor,
    /// Small constant added to the variance for numerical stability.
    pub eps: f32,
}

impl LayerNorm {
    /// Creates a `LayerNorm` for vectors of size `d_model`.
    #[must_use]
    pub fn new(d_model: usize) -> Self {
        Self {
            gamma: Tensor::ones(&[d_model]).with_grad(),
            beta: Tensor::zeros(&[d_model]).with_grad(),
            eps: 1e-5,
        }
    }

    /// Forward: `[S, D] → [S, D]`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        ops::layer_norm(x, &self.gamma, &self.beta, self.eps)
    }

    /// Returns all learnable parameters.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}
