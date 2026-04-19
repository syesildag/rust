//! ResNet residual block and backbone used in the CNN encoder.
//!
//! Each `ResidualBlock` applies:
//! ```text
//! Conv(3×3) → BN → ReLU → Conv(3×3) → BN → Add(input) → ReLU
//! ```
//! The skip connection prevents vanishing gradients in deep networks and lets
//! the block learn a *correction* on top of the identity transform.

use tensor::nn::{BatchNorm2d, Conv2d};
use tensor::ops;
use tensor::Tensor;

/// A single pre-activation residual block: same input/output shape `[N, C, H, W]`.
pub struct ResidualBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
}

impl ResidualBlock {
    /// Creates a residual block for `channels` feature maps with 3×3 kernels.
    #[must_use]
    pub fn new(channels: usize) -> Self {
        Self {
            conv1: Conv2d::new(channels, channels, 3, 1),
            bn1: BatchNorm2d::new(channels),
            conv2: Conv2d::new(channels, channels, 3, 1),
            bn2: BatchNorm2d::new(channels),
        }
    }

    /// Forward: `[N, C, H, W] → [N, C, H, W]`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = ops::relu(&self.bn1.forward(&self.conv1.forward(x)));
        let h = self.bn2.forward(&self.conv2.forward(&h));
        ops::relu(&ops::add(&h, x)) // residual skip connection
    }

    /// Returns all learnable parameters.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.conv1.parameters();
        p.extend(self.bn1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.bn2.parameters());
        p
    }
}

/// Stacked ResNet backbone: initial conv + N residual blocks.
///
/// Input `[N, 17, 8, 8]` → output `[N, 256, 8, 8]`.
pub struct ResNetBackbone {
    conv_in: Conv2d,
    bn_in: BatchNorm2d,
    blocks: Vec<ResidualBlock>,
}

impl ResNetBackbone {
    /// Creates a ResNet backbone.
    ///
    /// - `in_channels`: 17 (piece planes + metadata)
    /// - `channels`:    256 (feature maps)
    /// - `num_blocks`:  8 (residual blocks)
    #[must_use]
    pub fn new(in_channels: usize, channels: usize, num_blocks: usize) -> Self {
        Self {
            conv_in: Conv2d::new(in_channels, channels, 3, 1),
            bn_in: BatchNorm2d::new(channels),
            blocks: (0..num_blocks)
                .map(|_| ResidualBlock::new(channels))
                .collect(),
        }
    }

    /// Forward: `[N, 17, 8, 8] → [N, 256, 8, 8]`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = ops::relu(&self.bn_in.forward(&self.conv_in.forward(x)));
        self.blocks.iter().fold(h, |acc, block| block.forward(&acc))
    }

    /// Returns all learnable parameters.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.conv_in.parameters();
        p.extend(self.bn_in.parameters());
        for b in &self.blocks {
            p.extend(b.parameters());
        }
        p
    }
}
