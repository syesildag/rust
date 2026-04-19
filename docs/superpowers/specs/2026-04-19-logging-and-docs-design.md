# Logging & Documentation Design
**Date:** 2026-04-19  
**Status:** Approved  
**Scope:** All 5 crates — `chess`, `tensor`, `engine`, `cli`, `core`

---

## Goal

Add structured logging with performance profiling and comprehensive documentation to the Rust chess engine workspace. No existing logging infrastructure exists; all I/O is raw `println!`/`eprintln!`.

---

## 1. Logging Infrastructure

### Dependencies

Add to workspace `Cargo.toml` `[workspace.dependencies]`:
```toml
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

Add to each lib crate (`chess`, `tensor`, `engine`) `[dependencies]`:
```toml
tracing.workspace = true
```

Add to `cli` `[dependencies]`:
```toml
tracing.workspace = true
tracing-subscriber.workspace = true
```

### Subscriber Initialization (`cli/main.rs`)

```rust
tracing_subscriber::fmt()
    .with_env_filter(EnvFilter::from_default_env())
    .with_target(true)
    .with_elapsed_time(true)
    .init();
```

Initialized once at the top of `main()`, before any subcommand dispatch.

### Usage Convention

- `RUST_LOG=info` — training progress, eval results, game counts (default for users)
- `RUST_LOG=debug` — batch losses, move counts, FEN/PGN parse details
- `RUST_LOG=trace` — per-move evaluation, tensor shapes (disabled by default, expensive)

---

## 2. Span & Event Placement

### `cli/main.rs`
- `info_span!("train")` / `info_span!("eval")` / `info_span!("selfplay")` wrapping each subcommand
- Fields: command name, key config params (epochs, batch_size, lr)

### `engine/train.rs`
- `info_span!("epoch", epoch = i, total = config.epochs)` per epoch iteration
- `info!` event at epoch end: `loss`, elapsed time
- `debug!` event per batch: `batch`, `batch_loss`
- Replace existing `println!` epoch summary with `info!`

### `engine/selfplay.rs`
- `info_span!("selfplay", total_games = num_games)` wrapping generation loop
- `debug!` event per game: `game`, `outcome`, `plies`
- Replace existing `println!` progress every 10 games with `info!`

### `engine/dataset.rs`
- `info!` event after loading each file: `path`, `samples` count
- `info!` event after `from_pgn_files` completes: total samples across all files

### `engine/pgn.rs`
- `warn!` on skipped games (parse failure): include move that failed
- `warn!` on skipped individual moves within a game

### `chess/fen.rs`
- `warn!` on `FenError` with the offending FEN string fragment

### `chess/movegen.rs`
- `debug_span!("perft", depth)` wrapping `perft()` call
- `debug!` result: node count at each depth

### `engine/model.rs`
- `trace_span!("forward")` around `HybridValueNet::forward()` — disabled at `info`/`debug` level

---

## 3. Documentation

### Public API Docs (`///`)

Every public `struct`, `enum`, `fn`, `trait`, and `type alias` gets a `///` doc comment. Minimum: one sentence describing what it does and its key invariants or constraints.

**`chess/` crate:**
- `Board` — fields, invariants (12 bitboards layout), `make_move` side effects
- `Move`, `MoveKind` — when each variant applies
- `Square` — 0–63 index convention, file/rank layout
- Bitboard ops: `set_bit`, `clear_bit`, `lsb_square`, `pop_lsb`, `count_bits`
- `generate_legal_moves`, `generate_pseudo_legal`, `perft`
- `game_status`, `from_fen`, `to_fen`
- All attack functions: `knight_attacks`, `king_attacks`, `pawn_attacks`, `rook_attacks`, `bishop_attacks`, `queen_attacks`, `all_attacks`

**`tensor/` crate:**
- `Tensor` struct and all public methods
- All 30+ ops in `ops.rs` — each function documents shape contract (e.g., "inputs must have identical shapes")
- All `nn/` layers: `Linear`, `Conv2d`, `BatchNorm2d`, `LayerNorm`, `MultiHeadAttention`, `TransformerBlock`, `TransformerEncoder`
- `Adam` optimizer

**`engine/` crate:**
- `HybridValueNet` — architecture summary in struct doc
- `encode`, `encode_batch` — input/output shapes
- `ChessDataset`, `Sample`
- `train`, `TrainConfig`
- `generate` (self-play)

**`cli/` crate:**
- Module-level `//!` describing all subcommands and flags

### Architecture Docs (`//!` module headers)

**`engine/encode.rs`:**
- Full description of the 17-channel layout:
  - Channels 0–5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
  - Channels 6–11: Black pieces (same order)
  - Channel 12: Side to move (all 1s = White, all 0s = Black)
  - Channels 13–16: Castling rights (WK, WQ, BK, BQ)
- Why this representation: unambiguous, CNN-friendly, standard in AlphaZero-style engines

**`engine/model.rs`:**
- Pipeline: Board → 17-plane tensor → ResNet (8 blocks, 256 channels) → flatten to 64 tokens → prepend CLS → TransformerEncoder (4 blocks) → extract CLS → Linear(256→1) → tanh
- Why CLS token: aggregates global board context without positional bias
- Why tanh output: bounds score to (-1, +1), maps naturally to White/Black advantage

**`chess/attack.rs`:**
- Ray-casting via 8 direction index pairs `(file_delta, rank_delta)`
- Sliding pieces extend until they hit an occupied square (captures) or board edge
- Why bitboards: O(1) intersection checks with `&` operator

**`chess/movegen.rs`:**
- Pseudo-legal generation: fast, produces all moves ignoring whether king is left in check
- Legal filter: apply each pseudo-legal move, test `is_in_check()`, discard if king in check
- Why this two-pass approach: simpler than staged generation, correct by construction

---

## 4. Out of Scope

- GPU/wgpu logging (separate concern)
- `core` crate documentation (trivial `add` function, minimal value)
- Log file rotation or structured JSON output (overkill for a CLI tool)
- Benchmark harness (separate from logging)

---

## 5. Files Changed

| File | Change |
|------|--------|
| `Cargo.toml` (workspace) | Add `tracing`, `tracing-subscriber` to `[workspace.dependencies]` |
| `crates/chess/Cargo.toml` | Add `tracing` dependency |
| `crates/tensor/Cargo.toml` | Add `tracing` dependency |
| `crates/engine/Cargo.toml` | Add `tracing` dependency |
| `crates/cli/Cargo.toml` | Add `tracing`, `tracing-subscriber` dependencies |
| `crates/cli/src/main.rs` | Init subscriber; add command spans; keep user-facing `println!` |
| `crates/engine/src/train.rs` | Replace `println!` with `info!`/`debug!`; add epoch/batch spans |
| `crates/engine/src/selfplay.rs` | Replace `println!` with `info!`/`debug!`; add selfplay span |
| `crates/engine/src/dataset.rs` | Add `info!` on file load |
| `crates/engine/src/pgn.rs` | Add `warn!` on parse failures |
| `crates/chess/src/fen.rs` | Add `warn!` on FEN parse errors |
| `crates/chess/src/movegen.rs` | Add `debug_span!` on `perft` |
| `crates/engine/src/model.rs` | Add `trace_span!` on `forward`; add `//!` architecture doc |
| `crates/engine/src/encode.rs` | Add `//!` encoding doc; `///` on `encode`/`encode_batch` |
| `crates/chess/src/attack.rs` | Add `//!` algorithm doc; `///` on all attack functions |
| `crates/chess/src/movegen.rs` | Add `//!` movegen doc; `///` on public functions |
| `crates/chess/src/board.rs` | `///` on all public items |
| `crates/chess/src/piece.rs` | `///` on all public items |
| `crates/chess/src/square.rs` | `///` on all public items |
| `crates/chess/src/moves.rs` | `///` on all public items |
| `crates/chess/src/game.rs` | `///` on all public items |
| `crates/chess/src/fen.rs` | `///` on all public items |
| `crates/chess/src/bitboard.rs` | `///` on all public items |
| `crates/tensor/src/tensor_impl.rs` | `///` on all public items |
| `crates/tensor/src/ops.rs` | `///` on all 30+ ops with shape contracts |
| `crates/tensor/src/nn/*.rs` | `///` on all layers |
| `crates/tensor/src/optim.rs` | `///` on `Adam` |
| `crates/engine/src/dataset.rs` | `///` on all public items |
| `crates/engine/src/pgn.rs` | `///` on all public items |
| `crates/engine/src/train.rs` | `///` on all public items |
| `crates/engine/src/selfplay.rs` | `///` on all public items |
