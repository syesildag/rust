//! Batch normalisation for 4-D tensors `[N, C, H, W]`.

use crate::ops;
use crate::Tensor;

/// Normalises each channel over the N×H×W dimensions.
/// Learnable `gamma` (scale) and `beta` (shift), both `[C]`.
pub struct BatchNorm2d {
    /// Learnable per-channel scale parameter, shape `[C]`.
    pub gamma: Tensor,
    /// Learnable per-channel shift parameter, shape `[C]`.
    pub beta: Tensor,
    /// Small constant added to the variance for numerical stability.
    pub eps: f32,
}

impl BatchNorm2d {
    /// Creates a `BatchNorm2d` for `num_features` channels.
    #[must_use]
    pub fn new(num_features: usize) -> Self {
        Self {
            gamma: Tensor::ones(&[num_features]).with_grad(),
            beta: Tensor::zeros(&[num_features]).with_grad(),
            eps: 1e-3,
        }
    }

    /// Forward: `[N, C, H, W] → [N, C, H, W]`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        ops::batch_norm_2d(x, &self.gamma, &self.beta, self.eps)
    }

    /// Returns all learnable parameters.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}
