//! Supervised training loop for `HybridValueNet`.
//!
//! Uses MSE loss against game-outcome labels {+1, 0, -1} and the Adam optimizer.

use crate::dataset::ChessDataset;
use crate::model::HybridValueNet;
use std::path::PathBuf;
use tensor::optim::Adam;
use tensor::{ops, Tensor};
use tracing::{debug, info, info_span};

/// Hyper-parameters for a training run.
pub struct TrainConfig {
    /// PGN files (or directories of `.pgn` files) used as the training corpus.
    pub pgn_paths: Vec<PathBuf>,
    /// Number of full passes over the dataset.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Adam learning rate.
    pub lr: f32,
    /// Reserved for future checkpoint saving.
    pub output: PathBuf,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            pgn_paths: vec![PathBuf::from("games.pgn")],
            epochs: 20,
            batch_size: 32,
            lr: 1e-4,
            output: PathBuf::from("model.bin"),
        }
    }
}

/// Runs the full training loop.
///
/// Loads PGN games, shuffles them each epoch, computes MSE loss over each
/// mini-batch, and updates the model with Adam.
///
/// # Errors
/// Returns an error if no positions could be loaded (empty or unreadable PGN paths).
pub fn train(cfg: TrainConfig) -> Result<HybridValueNet, std::io::Error> {
    let model = HybridValueNet::new();
    let mut dataset = ChessDataset::from_pgn_files(&cfg.pgn_paths);
    let mut adam = Adam::new(model.parameters(), cfg.lr);

    if dataset.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "no positions loaded — check that the PGN paths are correct",
        ));
    }

    info!(
        positions = dataset.len(),
        epochs = cfg.epochs,
        batch_size = cfg.batch_size,
        "starting training"
    );

    for epoch in 0..cfg.epochs {
        let epoch_span = info_span!("epoch", n = epoch + 1, total = cfg.epochs);
        let _epoch_guard = epoch_span.enter();

        dataset.shuffle(epoch as u64);
        let mut total_loss = 0.0f32;
        let mut n_batches = 0usize;

        for batch in dataset.batches(cfg.batch_size) {
            adam.zero_grad();

            // Build summed MSE loss over the batch as one computation graph.
            let batch_loss = batch
                .iter()
                .map(|(board, label)| {
                    let pred = model.forward(board);
                    let target = Tensor::from_vec(vec![*label], &[1, 1]);
                    let diff = ops::sub(&pred, &target);
                    let d_val = diff.data()[0];
                    ops::mul_scalar(&diff, d_val) // diff * diff ≈ diff²
                })
                .reduce(|a, b| ops::add(&a, &b))
                .expect("batch is non-empty");

            let scale = 1.0 / batch.len() as f32;
            let loss = ops::mul_scalar(&batch_loss, scale);

            loss.backward();
            adam.step();

            total_loss += loss.data()[0];
            n_batches += 1;
            debug!(batch = n_batches, loss = loss.data()[0], "batch");
        }

        let avg = if n_batches > 0 {
            total_loss / n_batches as f32
        } else {
            0.0
        };
        info!(avg_loss = avg, "epoch complete");
    }

    model.save(&cfg.output)?;

    Ok(model)
}
