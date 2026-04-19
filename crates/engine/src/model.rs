//! `HybridValueNet`: ResNet backbone + Transformer head for chess position evaluation.
//!
//! ## Data flow
//! ```text
//! Board
//!   → encode()          [1, 17, 8, 8]
//!   → ResNetBackbone    [1, 256, 8, 8]
//!   → reshape           [64, 256]       (each square = one token)
//!   → prepend CLS       [65, 256]
//!   → + pos_embed       [65, 256]
//!   → TransformerEncoder[65, 256]
//!   → row 0 (CLS)       [256]
//!   → Linear(256→1)
//!   → tanh              ∈ (-1, +1)
//! ```

use crate::encode::encode_batch;
use crate::nn::ResNetBackbone;
use chess::board::Board;
use tensor::nn::{Linear, TransformerEncoder};
use tensor::{ops, Tensor};

const D_MODEL: usize = 256;
const NUM_HEADS: usize = 8;
const D_FF: usize = 1024;
const NUM_BLOCKS: usize = 4;
const IN_CHANNELS: usize = 17;
const CHANNELS: usize = 256;
const NUM_RES: usize = 8;
/// Sequence length = 64 squares + 1 CLS token.
const SEQ_LEN: usize = 65;

/// Chess hybrid value network combining a ResNet CNN backbone with a Transformer encoder.
///
/// Outputs a scalar in (-1, +1): positive = White advantage, negative = Black advantage.
pub struct HybridValueNet {
    backbone: ResNetBackbone,
    /// Learnable CLS token `[1, 256]`.
    cls_token: Tensor,
    /// Learnable positional embeddings `[65, 256]`.
    pos_embed: Tensor,
    encoder: TransformerEncoder,
    head: Linear,
}

impl HybridValueNet {
    /// Creates a new randomly-initialised `HybridValueNet`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backbone: ResNetBackbone::new(IN_CHANNELS, CHANNELS, NUM_RES),
            cls_token: Tensor::randn(&[1, D_MODEL], 0.02).with_grad(),
            pos_embed: Tensor::randn(&[SEQ_LEN, D_MODEL], 0.02).with_grad(),
            encoder: TransformerEncoder::new(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF),
            head: Linear::new(D_MODEL, 1, true),
        }
    }

    /// Evaluates a board position.
    ///
    /// Returns a scalar tensor with value ∈ (-1, +1).
    #[must_use]
    pub fn forward(&self, board: &Board) -> Tensor {
        // 1. Encode → [1, 17, 8, 8]
        let x = encode_batch(board);

        // 2. CNN backbone → [1, 256, 8, 8]
        let x = self.backbone.forward(&x);

        // 3. Reshape [1, 256, 8, 8] → [64, 256]
        //    Each of the 64 squares becomes one 256-dim token.
        let x = x.reshape(&[64, D_MODEL]);

        // 4. Prepend CLS token → [65, 256]
        let x = ops::cat(&[&self.cls_token, &x]);

        // 5. Add positional embeddings (learned)
        let x = ops::add(&x, &self.pos_embed);

        // 6. Transformer encoder → [65, 256]
        let x = self.encoder.forward(&x);

        // 7. Extract CLS (row 0) → [256], then project to scalar
        let cls = ops::select_row(&x, 0);
        ops::tanh(&self.head.forward(&cls))
    }

    /// Collects all learnable parameters (for the optimizer).
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.backbone.parameters();
        p.push(self.cls_token.clone());
        p.push(self.pos_embed.clone());
        p.extend(self.encoder.parameters());
        p.extend(self.head.parameters());
        p
    }
}

impl Default for HybridValueNet {
    fn default() -> Self {
        Self::new()
    }
}
