//! A minimal ML framework with CPU and Metal (wgpu) acceleration.
//!
//! Provides [`Tensor`], automatic differentiation, neural network layers,
//! and the Adam optimizer.
//!
//! # Quick start
//! ```
//! use tensor::{Tensor, ops};
//! use tensor::nn::Linear;
//!
//! let x = Tensor::randn(&[4, 8], 0.02);
//! let layer = Linear::new(8, 4, true);
//! let y = layer.forward(&x);
//! assert_eq!(y.shape(), &[4, 4]);
//! ```

#![allow(clippy::module_name_repetitions)]

pub mod gpu;
pub mod nn;
pub mod ops;
pub mod optim;
mod tensor_impl;

pub use gpu::GpuContext;
pub use tensor_impl::{GradFn, Tensor};
