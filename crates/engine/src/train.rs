//! Supervised training loop for `HybridValueNet`.
//!
//! Uses MSE loss against game-outcome labels {+1, 0, -1} and the Adam optimizer.

use crate::dataset::ChessDataset;
use crate::model::HybridValueNet;
use crate::persist::Persist;
use crate::position_db::PositionDb;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tensor::optim::Adam;
use tensor::{ops, Tensor};
use tracing::{info, info_span, warn};

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
    /// Path where the trained model is written.
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
/// Positions tracked in the `.posdb` sidecar file are skipped when
/// `stored_epoch > current_epoch`, enabling resume after interruption.
/// Send SIGTERM or SIGINT to trigger a clean save before exit.
///
/// # Errors
/// Returns an error if no positions could be loaded (empty or unreadable PGN paths).
pub fn train(cfg: TrainConfig) -> Result<HybridValueNet, std::io::Error> {
    let model = HybridValueNet::load_from(&cfg.output)
        .inspect(|_| info!(path = %cfg.output.display(), "restored model weights"))
        .inspect_err(|e| warn!(error = %e, "ignoring saved model — starting fresh"))
        .unwrap_or_default();
    let mut dataset = ChessDataset::load_with_cache(&cfg.pgn_paths);

    if dataset.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "no positions loaded — check that the PGN paths are correct",
        ));
    }

    let db_path = cfg.output.with_extension("pos");
    let mut pos_db = PositionDb::load_from(&db_path)
        .inspect(|_| info!(path = %db_path.display(), "restored position db"))
        .inspect_err(|e| warn!(error = %e, "ignoring saved position db — starting fresh"))
        .unwrap_or_default();

    let loss_path = cfg.output.with_file_name("loss.csv");

    let adam_path = cfg.output.with_extension("adam");
    let mut adam = Adam::load_from(&adam_path)
        .and_then(|saved| saved.with_params(model.parameters(), cfg.lr))
        .inspect(|_| info!(path = %adam_path.display(), "restored Adam optimizer state"))
        .inspect_err(|e| warn!(error = %e, "ignoring saved Adam state — starting fresh"))
        .unwrap_or_else(|_| Adam::new(model.parameters(), cfg.lr));

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_flag = Arc::clone(&shutdown);
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .build()
            .expect("tokio signal runtime");
        rt.block_on(async {
            if tokio::signal::ctrl_c().await.is_ok() {
                shutdown_flag.store(true, Ordering::SeqCst);
            }
        });
    });

    info!(
        positions = dataset.len(),
        epochs = cfg.epochs,
        batch_size = cfg.batch_size,
        "starting training"
    );

    let mut loss_log: Vec<(usize, f32, f32)> = Vec::new();

    'training: for epoch in 1..=cfg.epochs {
        let epoch_span = info_span!("epoch", n = epoch, total = cfg.epochs);
        let _epoch_guard = epoch_span.enter();

        dataset.shuffle(epoch as u64);

        let mut n_samples = 0usize;

        for batch in dataset.batches(cfg.batch_size) {
            if shutdown.load(Ordering::SeqCst) {
                info!("shutdown signal received — saving progress and exiting…");
                break 'training;
            }

            let filtered: Vec<_> = batch
                .iter()
                .filter(|(board, _, game_id)| !pos_db.should_skip(&board.to_fen(), *game_id, epoch))
                .collect();

            n_samples += batch.len();

            if filtered.is_empty() {
                continue;
            }

            adam.zero_grad();

            let (boards, outcomes): (Vec<_>, Vec<f32>) =
                filtered.iter().map(|(b, l, _)| ((*b).clone(), *l)).unzip();
            let b = boards.len();

            let preds = model.forward_batch(&boards);
            let targets = Tensor::from_vec(outcomes, &[b, 1]);
            let loss = ops::mse_loss_tensor(&preds, &targets);

            let loss_val = loss.data()[0];
            if !loss_val.is_finite() {
                warn!(
                    loss = loss_val,
                    "non-finite loss — skipping optimizer step for this batch"
                );
                continue;
            }

            loss.backward();
            adam.clip_grad_norm(1.0);
            adam.step();

            let pct = n_samples as f32 / dataset.len() as f32 * 100.0;
            info!(
                percentage = format!("{pct:.1}%"),
                loss = loss_val
            );
            loss_log.push((epoch, pct, loss_val));

            for (board, _, game_id) in &filtered {
                pos_db.record(&board.to_fen(), *game_id, epoch);
            }
        }

        info!("epoch {epoch} complete");
    }

    model.save_to(&cfg.output)?;
    pos_db.save_to(&db_path)?;
    adam.save_to(&adam_path)?;
    append_loss_log(&loss_path, &loss_log)?;
    Ok(model)
}

fn append_loss_log(
    path: &std::path::Path,
    entries: &[(usize, f32, f32)],
) -> std::io::Result<()> {
    use std::io::Write;
    let needs_header = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if needs_header {
        writeln!(file, "epoch,percentage,loss")?;
    }
    for &(epoch, pct, loss) in entries {
        writeln!(file, "{epoch},{pct:.1},{loss:.8}")?;
    }
    Ok(())
}
