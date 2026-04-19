//! 2-D convolutional layer.

#![allow(clippy::cast_precision_loss)]

use crate::ops;
use crate::Tensor;

/// A 2-D convolutional layer: `output = conv2d(input, weight, bias, padding)`.
///
/// - `weight`: `[C_out, C_in, kH, kW]`
/// - `bias`:   `[C_out]`
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub padding: usize,
}

impl Conv2d {
    /// Creates a `Conv2d` layer.
    ///
    /// Weights are He-initialised; bias is zero.
    #[must_use]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
    ) -> Self {
        let fan_in = in_channels * kernel_size * kernel_size;
        let std = (2.0_f32 / fan_in as f32).sqrt();
        Self {
            weight: Tensor::randn(&[out_channels, in_channels, kernel_size, kernel_size], std)
                .with_grad(),
            bias: Tensor::zeros(&[out_channels]).with_grad(),
            padding,
        }
    }

    /// Forward: `[N, C_in, H, W] → [N, C_out, H_out, W_out]`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        ops::conv2d(x, &self.weight, &self.bias, self.padding)
    }

    /// Returns all learnable parameters.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
