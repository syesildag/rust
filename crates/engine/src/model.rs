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
//!
//! ## CLS token
//!
//! The CLS ("classification") token is a learnable vector prepended to the
//! 64 square tokens before the transformer encoder. By the final layer, it has
//! attended to all 64 squares and aggregates global board context. Extracting
//! only the CLS output (row 0) gives a fixed-size representation suitable for
//! the scalar head, without any positional bias.
//!
//! ## tanh output
//!
//! The final `tanh` bounds the output to (-1, +1), matching the training labels
//! (+1.0 = White wins, -1.0 = Black wins, 0.0 = draw). Bounded output also
//! stabilises MSE loss by preventing divergence during early training.

use crate::encode::{encode_batch, encode_boards};
use crate::nn::ResNetBackbone;
use crate::persist::Persist;
use chess::board::Board;
use std::io::{Read, Write};
use tensor::nn::{Linear, TransformerEncoder};
use tensor::{ops, Tensor};
use tracing::trace_span;

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
    /// ResNet CNN backbone extracting spatial features from the 17-plane board tensor.
    backbone: ResNetBackbone,
    /// Learnable CLS token prepended to the 64 square tokens; shape `[1, 256]`.
    cls_token: Tensor,
    /// Learnable positional embeddings for all 65 tokens (CLS + 64 squares); shape `[65, 256]`.
    pos_embed: Tensor,
    /// Transformer encoder that mixes information across all 65 tokens.
    encoder: TransformerEncoder,
    /// Linear projection from the 256-dim CLS embedding to a scalar evaluation.
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
        let _span = trace_span!("HybridValueNet::forward").entered();
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

    /// Evaluates a batch of board positions.
    ///
    /// Returns `[B, 1]` with values ∈ (-1, +1).
    /// The CNN backbone processes all boards simultaneously; the transformer is applied
    /// per item so that the existing single-sequence attention code requires no change.
    #[must_use]
    pub fn forward_batch(&self, boards: &[Board]) -> Tensor {
        let _span = trace_span!("HybridValueNet::forward_batch").entered();
        let b = boards.len();

        // 1. Encode all boards → [B, 17, 8, 8]
        let x = encode_boards(boards);

        // 2. CNN backbone → [B, 256, 8, 8]
        let x = self.backbone.forward(&x);

        // 3. Build [B, 65, 256]: prepend CLS token + add positional embeddings per item.
        let mut seqs: Vec<Tensor> = Vec::with_capacity(b);
        for i in 0..b {
            let xi = ops::slice_batch(&x, i).reshape(&[64, D_MODEL]); // [64, 256]
            let xi = ops::cat(&[&self.cls_token, &xi]);                // [65, 256]
            seqs.push(ops::add(&xi, &self.pos_embed));
        }
        let seq_refs: Vec<&Tensor> = seqs.iter().collect();
        let x = ops::stack(&seq_refs); // [B, 65, 256]

        // 4. Batched transformer → [B, 65, 256]
        let x = self.encoder.forward_batched(&x, b);

        // 5. Extract CLS (row 0) per item → stack → [B, 256]
        let mut cls_outputs: Vec<Tensor> = Vec::with_capacity(b);
        for i in 0..b {
            cls_outputs.push(ops::select_row(&ops::slice_batch(&x, i), 0));
        }
        let cls_refs: Vec<&Tensor> = cls_outputs.iter().collect();
        let x = ops::cat(&cls_refs); // [B, 256]

        // 6. Project → [B, 1], tanh → [B, 1]
        ops::tanh(&self.head.forward(&x))
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

/// Binary format: `[u64 num_params]` then for each param:
/// `[u64 ndim] [u64; ndim shape] [f32; numel data]`
impl Persist for HybridValueNet {
    fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let params = self.parameters();
        w.write_all(&(params.len() as u64).to_le_bytes())?;
        for p in &params {
            let shape = p.shape();
            w.write_all(&(shape.len() as u64).to_le_bytes())?;
            for &dim in shape {
                w.write_all(&(dim as u64).to_le_bytes())?;
            }
            for &val in &p.data() {
                w.write_all(&val.to_le_bytes())?;
            }
        }
        Ok(())
    }

    #[allow(clippy::cast_possible_truncation)]
    fn read_from<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf8)?;
        let num_params = u64::from_le_bytes(buf8) as usize;

        let model = Self::new();
        let params = model.parameters();
        if params.len() != num_params {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "parameter count mismatch: file has {num_params}, model expects {}",
                    params.len()
                ),
            ));
        }

        for p in &params {
            r.read_exact(&mut buf8)?;
            let ndim = u64::from_le_bytes(buf8) as usize;
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                r.read_exact(&mut buf8)?;
                shape.push(u64::from_le_bytes(buf8) as usize);
            }
            if shape != p.shape() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("shape mismatch: file has {shape:?}, model expects {:?}", p.shape()),
                ));
            }
            let numel: usize = shape.iter().product();
            let mut data = Vec::with_capacity(numel);
            for _ in 0..numel {
                r.read_exact(&mut buf4)?;
                data.push(f32::from_le_bytes(buf4));
            }
            p.set_data(&data);
        }
        Ok(model)
    }
}

impl Default for HybridValueNet {
    fn default() -> Self {
        Self::new()
    }
}
