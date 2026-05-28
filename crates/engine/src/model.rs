//! `HybridValueNet`: ResNet backbone + Transformer head for chess position evaluation.
//!
//! ## Data flow
//! ```text
//! Board
//!   → encode()          [1, 18, 8, 8]
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
use tensor::nn::{LayerNorm, Linear, TransformerEncoder};
use tensor::{ops, Tensor};
use tracing::{trace_span, warn};

const D_MODEL: usize = 256;
const NUM_HEADS: usize = 8;
const D_FF: usize = 1024;
const NUM_BLOCKS: usize = 4;
const IN_CHANNELS: usize = 18;
const CHANNELS: usize = 256;
const NUM_RES: usize = 8;
/// Dropout probability applied in each transformer block.
const DROPOUT: f32 = 0.1;
/// Sequence length = 64 squares + 1 CLS token.
const SEQ_LEN: usize = 65;

fn nonfinite_debug_enabled() -> bool {
    std::env::var("ENGINE_DEBUG_NONFINITE")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn log_nonfinite_stage(stage: &str, t: &Tensor, boards: &[Board], enabled: bool) {
    if !enabled {
        return;
    }

    let shape = t.shape().to_vec();
    let data = t.data();
    let n_nonfinite = data.iter().filter(|v| !v.is_finite()).count();
    if n_nonfinite == 0 {
        return;
    }

    let batch = shape.first().copied().unwrap_or(0);
    let batch_stride = if batch > 0 { data.len() / batch } else { 0 };
    let examples: Vec<String> = data
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_finite())
        .take(3)
        .map(|(flat_idx, &val)| {
            let batch_idx = if batch_stride > 0 {
                flat_idx / batch_stride
            } else {
                0
            };
            let fen = boards
                .get(batch_idx)
                .map_or_else(|| "<unknown>".to_string(), Board::to_fen);
            format!("flat={flat_idx} batch={batch_idx} val={val} fen={fen}")
        })
        .collect();

    warn!(
        stage,
        shape = ?shape,
        n_nonfinite,
        examples = ?examples,
        "non-finite activations detected"
    );
}

/// Chess hybrid value network combining a ResNet CNN backbone with a Transformer encoder.
///
/// Outputs a scalar in (-1, +1): positive = White advantage, negative = Black advantage.
pub struct HybridValueNet {
    /// ResNet CNN backbone extracting spatial features from the 17-plane board tensor.
    backbone: ResNetBackbone,
    /// Layer norm applied to backbone tokens before the transformer encoder.
    /// The backbone output is relu-clipped and batch-normalised per channel but
    /// NOT normalised per token; without this norm the first transformer block
    /// sees activations in [0, ~10], producing attention scores in the hundreds
    /// that destabilise training even after score clamping.
    pre_norm: LayerNorm,
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
            pre_norm: LayerNorm::new(D_MODEL),
            cls_token: Tensor::randn(&[1, D_MODEL], 0.02).with_grad(),
            pos_embed: Tensor::randn(&[SEQ_LEN, D_MODEL], 0.02).with_grad(),
            encoder: TransformerEncoder::new(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, DROPOUT),
            head: Linear::new(D_MODEL, 1, true),
        }
    }

    /// Switches between training (`true`) and inference (`false`) mode.
    ///
    /// Must be set to `false` during self-play and evaluation to disable dropout.
    pub fn set_training(&self, training: bool) {
        self.encoder.set_training(training);
    }

    /// Evaluates a board position.
    ///
    /// Returns a scalar tensor with value ∈ (-1, +1).
    #[must_use]
    pub fn forward(&self, board: &Board) -> Tensor {
        let _span = trace_span!("HybridValueNet::forward").entered();
        // 1. Encode → [1, 18, 8, 8]
        let x = encode_batch(board);

        // 2. CNN backbone → [1, 256, 8, 8]
        let x = self.backbone.forward(&x);

        // 3. Reshape [1, 256, 8, 8] → [64, 256]
        //    Each of the 64 squares becomes one 256-dim token.
        let x = x.reshape(&[64, D_MODEL]);

        // 3b. Normalise per token before the transformer sees them.
        //     Backbone output is relu-clipped and batch-normalised per channel
        //     but not per token; without this the first block's attention scores
        //     can reach O(1000) and trigger NaN in subsequent operations.
        let x = self.pre_norm.forward(&x);

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
    /// All sequence construction (CLS prepend, positional embeddings) and CLS
    /// extraction are fully vectorised — no per-item loops.
    #[must_use]
    pub fn forward_batch(&self, boards: &[Board]) -> Tensor {
        let _span = trace_span!("HybridValueNet::forward_batch").entered();
        let b = boards.len();
        let debug_nonfinite = nonfinite_debug_enabled();

        // 1. Encode all boards → [B, 18, 8, 8]
        let x = encode_boards(boards);
        log_nonfinite_stage("encode_boards", &x, boards, debug_nonfinite);

        // 2. CNN backbone → [B, 256, 8, 8]
        let x = self.backbone.forward(&x);
        log_nonfinite_stage("backbone", &x, boards, debug_nonfinite);

        // 3. Reshape → [B, 64, 256]: each spatial position becomes one token.
        let x = x.reshape(&[b, 64, D_MODEL]);
        log_nonfinite_stage("reshape_tokens", &x, boards, debug_nonfinite);

        // 3b. Per-token normalisation before the transformer.
        //     Applied to [B*64, 256] then reshaped back.
        let x = self
            .pre_norm
            .forward(&x.reshape(&[b * 64, D_MODEL]))
            .reshape(&[b, 64, D_MODEL]);
        log_nonfinite_stage("pre_norm", &x, boards, debug_nonfinite);

        // 4. Prepend CLS → [B, 65, 256], then broadcast-add positional embeddings.
        let x = ops::prepend_cls_batched(&self.cls_token, &x);
        log_nonfinite_stage("prepend_cls", &x, boards, debug_nonfinite);
        let x = ops::broadcast_add_batch(&x, &self.pos_embed);
        log_nonfinite_stage("add_pos_embed", &x, boards, debug_nonfinite);

        // 5. Batched transformer → [B, 65, 256]
        let x = self.encoder.forward_batched(&x, b);
        log_nonfinite_stage("encoder_batched", &x, boards, debug_nonfinite);

        // 6. Extract CLS token (position 0) from every item → [B, 256]
        let x = ops::select_token(&x, 0);
        log_nonfinite_stage("select_cls", &x, boards, debug_nonfinite);

        // 7. Project → [B, 1], tanh → [B, 1]
        let x = self.head.forward(&x);
        log_nonfinite_stage("head_linear", &x, boards, debug_nonfinite);
        let x = ops::tanh(&x);
        log_nonfinite_stage("head_tanh", &x, boards, debug_nonfinite);
        x
    }

    /// Collects all learnable parameters (for the optimizer).
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.backbone.parameters();
        p.extend(self.pre_norm.parameters());
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
                    format!(
                        "shape mismatch: file has {shape:?}, model expects {:?}",
                        p.shape()
                    ),
                ));
            }
            let numel: usize = shape.iter().product();
            let mut data = Vec::with_capacity(numel);
            for _ in 0..numel {
                r.read_exact(&mut buf4)?;
                data.push(f32::from_le_bytes(buf4));
            }
            if data.iter().any(|v| !v.is_finite()) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "saved weights contain NaN/Inf — checkpoint is corrupt, delete and retrain",
                ));
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
