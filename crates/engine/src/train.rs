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
use tracing::{debug, info, info_span, warn};

/// Hyper-parameters for a training run.
pub struct TrainConfig {
    /// PGN files (or directories of `.pgn` files) used as the training corpus.
    pub pgn_paths: Vec<PathBuf>,
    /// Number of full passes over the dataset.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Base (minimum) Adam learning rate used at the start and end of the schedule.
    pub lr: f32,
    /// Peak learning rate reached at the end of warmup.
    /// If equal to `lr`, the schedule is disabled and `lr` is used as a constant.
    pub max_lr: f32,
    /// Number of optimizer steps over which the LR linearly increases from `lr` to
    /// `max_lr`. After warmup the LR follows a cosine decay back to `lr`. Set to
    /// `0` to skip warmup (cosine decay still applies if `max_lr > lr`).
    pub warmup_steps: usize,
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
            max_lr: 1e-4,
            warmup_steps: 0,
            output: PathBuf::from("model.bin"),
        }
    }
}

/// Computes the scheduled learning rate for the given optimizer step.
///
/// - Steps `0..warmup_steps`: linear ramp from `base_lr` → `max_lr`.
/// - Steps `warmup_steps..total_steps`: cosine decay from `max_lr` → `base_lr`.
fn scheduled_lr(step: usize, base_lr: f32, max_lr: f32, warmup_steps: usize, total_steps: usize) -> f32 {
    if step < warmup_steps {
        let progress = step as f32 / warmup_steps.max(1) as f32;
        base_lr + (max_lr - base_lr) * progress
    } else {
        let decay_steps = total_steps.saturating_sub(warmup_steps).max(1);
        let t = (step - warmup_steps) as f32;
        let progress = (t / decay_steps as f32).min(1.0);
        base_lr + 0.5 * (max_lr - base_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
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
#[allow(clippy::too_many_lines)]
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

    let step_sleep_ms = std::env::var("TRAIN_STEP_SLEEP_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    // Estimate total optimizer steps for the cosine-decay schedule.
    // The dataset is shuffled each epoch so the batch count is stable across epochs.
    let use_schedule = cfg.max_lr > cfg.lr;
    let batches_per_epoch = dataset.len().div_ceil(cfg.batch_size);
    let total_steps = cfg.epochs * batches_per_epoch;

    info!(
        positions = dataset.len(),
        epochs = cfg.epochs,
        batch_size = cfg.batch_size,
        lr = cfg.lr,
        max_lr = cfg.max_lr,
        warmup_steps = cfg.warmup_steps,
        total_steps,
        step_sleep_ms,
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

            let pct = n_samples as f32 / dataset.len() as f32 * 100.0;
            let loss_val = loss.data()[0];
            if !loss_val.is_finite() {
                let pred_data = preds.data();
                let finite_preds: Vec<f32> = pred_data
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect();
                let pred_min = finite_preds.iter().copied().fold(f32::INFINITY, f32::min);
                let pred_max = finite_preds
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let bad_positions: Vec<String> = pred_data
                    .iter()
                    .zip(boards.iter())
                    .filter_map(|(pred, board)| {
                        if pred.is_finite() {
                            None
                        } else {
                            Some(format!("pred={pred} fen={}", board.to_fen()))
                        }
                    })
                    .collect();
                let target_data = targets.data();
                let target_min = target_data.iter().copied().fold(f32::INFINITY, f32::min);
                let target_max = target_data
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                // Log the L2 norm of each parameter tensor.  An exploding norm
                // (e.g. > 100) points to the source layer of the instability.
                let param_norms: Vec<String> = model
                    .parameters()
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let norm: f32 = p.data().iter().map(|v| v * v).sum::<f32>().sqrt();
                        format!("p{i}:{norm:.3}")
                    })
                    .collect();
                warn!(
                    loss = loss_val,
                    percentage = format!("{pct:.1}%"),
                    batch_size = b,
                    pred_min,
                    pred_max,
                    n_nonfinite_preds = bad_positions.len(),
                    bad_positions = ?bad_positions,
                    target_min,
                    target_max,
                    param_norms = %param_norms.join(" "),
                    "non-finite loss — skipping optimizer step for this batch"
                );
                continue;
            }

            loss.backward();
            if !adam.clip_grad_norm(1.0) {
                // Find the first parameter whose gradient went non-finite to help
                // trace which layer is the source of the numerical instability.
                let first_bad = model
                    .parameters()
                    .iter()
                    .enumerate()
                    .find(|(_, p)| p.grad().iter().any(|g| !g.is_finite()))
                    .map(|(i, _)| i);
                warn!(
                    percentage = format!("{pct:.1}%"),
                    first_nonfinite_param = ?first_bad,
                    "non-finite gradients after backward — skipping optimizer step"
                );
                continue;
            }
            adam.step();

            // Apply warmup + cosine-decay schedule *after* the step so the
            // next iteration uses the updated LR.
            if use_schedule {
                let step = adam.state().0;
                let new_lr = scheduled_lr(step, cfg.lr, cfg.max_lr, cfg.warmup_steps, total_steps);
                adam.set_lr(new_lr);
            }

            if step_sleep_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(step_sleep_ms));
            }

            info!(percentage = format!("{pct:.1}%"), loss = loss_val, lr = adam.lr());
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

fn append_loss_log(path: &std::path::Path, entries: &[(usize, f32, f32)]) -> std::io::Result<()> {
    use std::io::Write;
    debug!(path = %path.display(), "saving");
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
