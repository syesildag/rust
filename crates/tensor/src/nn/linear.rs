//! Fully-connected linear layer: `output = input @ weight.T + bias`.

#![allow(clippy::cast_precision_loss)]

use crate::ops;
use crate::Tensor;

/// A fully-connected layer with learnable `weight` `[out, in]` and optional `bias` `[out]`.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    /// Creates a new `Linear` layer.
    ///
    /// Weights are initialised from N(0, 0.02²); bias is initialised to zero.
    ///
    /// # Panics
    /// Panics if `in_features` or `out_features` is zero.
    #[must_use]
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let std = (2.0_f32 / in_features as f32).sqrt(); // He init
        let weight = Tensor::randn(&[out_features, in_features], std).with_grad();
        let bias = if use_bias {
            Tensor::zeros(&[out_features]).with_grad()
        } else {
            Tensor::zeros(&[out_features])
        };
        Self { weight, bias }
    }

    /// Forward pass: `[S, in] → [S, out]`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        ops::linear(x, &self.weight, &self.bias)
    }

    /// Returns all learnable parameter tensors.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        if self.bias.requires_grad() {
            vec![self.weight.clone(), self.bias.clone()]
        } else {
            vec![self.weight.clone()]
        }
    }
}
