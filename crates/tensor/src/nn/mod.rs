//! Neural network building blocks.
//!
//! All layers implement a simple interface: they own their parameters as
//! `Tensor`s (with `requires_grad = true`) and expose a `forward` method
//! plus a `parameters` method that returns all leaf tensors.

pub mod attention;
pub mod batch_norm;
pub mod conv2d;
pub mod layer_norm;
pub mod linear;
pub mod transformer;

pub use attention::MultiHeadAttention;
pub use batch_norm::BatchNorm2d;
pub use conv2d::Conv2d;
pub use layer_norm::LayerNorm;
pub use linear::Linear;
pub use transformer::{TransformerBlock, TransformerEncoder};
