//! Dropout regularisation layer.
//!
//! Applies inverted dropout during training and passes the tensor through
//! unchanged during inference.  The training/inference mode is toggled via
//! [`Dropout::set_training`] and stored in a [`std::cell::Cell`] so that
//! the model can be switched without requiring `&mut self`.

use std::cell::Cell;

use crate::ops;
use crate::Tensor;

/// Inverted dropout layer.
///
/// During training each element is independently zeroed with probability `p`
/// and scaled by `1 / (1 - p)`.  During inference the layer is a no-op.
pub struct Dropout {
    /// Drop probability in `[0, 1)`.
    p: f32,
    /// `true` while training, `false` during inference.
    training: Cell<bool>,
}

impl Dropout {
    /// Creates a new `Dropout` layer with drop probability `p`.
    ///
    /// Starts in **training** mode.
    #[must_use]
    pub fn new(p: f32) -> Self {
        Self {
            p,
            training: Cell::new(true),
        }
    }

    /// Switches between training (`true`) and inference (`false`) mode.
    pub fn set_training(&self, v: bool) {
        self.training.set(v);
    }

    /// Forward pass: applies dropout when training, identity otherwise.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        ops::dropout(x, self.p, self.training.get())
    }

    /// Returns all learnable parameters (none — dropout has no parameters).
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
