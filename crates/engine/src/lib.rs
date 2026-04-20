//! Chess hybrid value network: ResNet backbone + Transformer head.
//!
//! Encodes board positions as `[17×8×8]` binary planes and evaluates them
//! with a value ∈ (-1, +1) — positive = White advantage.

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::doc_markdown)]

pub mod dataset;
mod position_db;
pub mod encode;
pub mod fen_file;
pub mod model;
pub mod nn;
pub mod pgn;
pub mod selfplay;
pub mod train;

pub use model::HybridValueNet;
pub use train::{train, TrainConfig};
