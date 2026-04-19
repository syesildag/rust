# Logging & Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tracing`-based structured logging with performance profiling and comprehensive `///`/`//!` documentation across all 5 crates.

**Architecture:** Add `tracing` to lib crates and `tracing-subscriber` to the CLI binary only. Replace all `println!`/`eprintln!` in training/selfplay/dataset with structured `info!`/`debug!`/`warn!` events. Add epoch/batch/selfplay spans for timing. Document every public item with `///` and add `//!` architecture headers to complex modules.

**Tech Stack:** `tracing = "0.1"`, `tracing-subscriber = { version = "0.3", features = ["env-filter"] }`

---

## Files Modified

| File | Change |
|------|--------|
| `Cargo.toml` | Add `[workspace.dependencies]` with tracing crates |
| `crates/chess/Cargo.toml` | Add `tracing.workspace = true` |
| `crates/tensor/Cargo.toml` | Add `tracing.workspace = true` |
| `crates/engine/Cargo.toml` | Add `tracing.workspace = true` |
| `crates/cli/Cargo.toml` | Add `tracing` + `tracing-subscriber` |
| `crates/cli/src/main.rs` | Init subscriber; add command spans |
| `crates/engine/src/train.rs` | Replace `println!` with `info!`/`debug!`; add spans |
| `crates/engine/src/selfplay.rs` | Replace `println!` with `info!`/`debug!`; add span |
| `crates/engine/src/dataset.rs` | Replace `println!`/`eprintln!` with `info!`/`warn!` |
| `crates/engine/src/pgn.rs` | Add `warn!` on skipped SAN tokens |
| `crates/engine/src/fen_file.rs` | Add `warn!` on FEN line parse failures |
| `crates/chess/src/movegen.rs` | Add `debug_span!` to `perft`; `//!` architecture doc; `///` on public items |
| `crates/engine/src/model.rs` | Add `trace_span!` to `forward`; `///` additions |
| `crates/chess/src/attack.rs` | `//!` architecture doc; `///` on all public items |
| `crates/chess/src/board.rs` | `///` on all public items |
| `crates/chess/src/piece.rs` | `///` on all public items |
| `crates/chess/src/square.rs` | `///` on all public items |
| `crates/chess/src/moves.rs` | `///` on all public items |
| `crates/chess/src/game.rs` | `///` on all public items |
| `crates/chess/src/fen.rs` | `///` on error variants |
| `crates/chess/src/bitboard.rs` | `///` on all public items |
| `crates/tensor/src/tensor_impl.rs` | `///` on all public items |
| `crates/tensor/src/ops.rs` | `///` on all 30+ ops with shape contracts |
| `crates/tensor/src/nn/*.rs` | `///` on all layer structs and methods |
| `crates/tensor/src/optim.rs` | `///` on `Adam` |
| `crates/engine/src/encode.rs` | `//!` architecture doc; `///` on `encode`/`encode_batch` |
| `crates/engine/src/dataset.rs` | `///` additions |
| `crates/engine/src/pgn.rs` | `///` on `Sample`, `parse_pgn` |

---

### Task 1: Add tracing dependencies

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/chess/Cargo.toml`
- Modify: `crates/tensor/Cargo.toml`
- Modify: `crates/engine/Cargo.toml`
- Modify: `crates/cli/Cargo.toml`

- [ ] **Step 1: Add workspace dependencies section to root Cargo.toml**

Add after the `[workspace.lints.clippy]` block:

```toml
[workspace.dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

- [ ] **Step 2: Add tracing to chess, tensor, engine crates**

In `crates/chess/Cargo.toml`, add:
```toml
[dependencies]
tracing.workspace = true
```

In `crates/tensor/Cargo.toml`, add after the existing `[dependencies]` entries:
```toml
tracing.workspace = true
```

In `crates/engine/Cargo.toml`, add after the existing `[dependencies]` entries:
```toml
tracing.workspace = true
```

- [ ] **Step 3: Add tracing + tracing-subscriber to cli**

In `crates/cli/Cargo.toml`, add:
```toml
tracing.workspace = true
tracing-subscriber.workspace = true
```

- [ ] **Step 4: Verify compilation**

```bash
cargo build --all
```
Expected: compiles cleanly with no errors or warnings.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock crates/chess/Cargo.toml crates/tensor/Cargo.toml crates/engine/Cargo.toml crates/cli/Cargo.toml
git commit -m "feat: add tracing dependencies to workspace"
```

---

### Task 2: Initialize tracing subscriber in CLI

**Files:**
- Modify: `crates/cli/src/main.rs`

- [ ] **Step 1: Add subscriber initialization**

In `crates/cli/src/main.rs`, add these imports at the top:
```rust
use tracing::info_span;
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};
```

Replace the opening of `fn main()`:
```rust
fn main() {
    let args: Vec<String> = std::env::args().collect();
```
with:
```rust
fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(true)
        .with_span_events(FmtSpan::CLOSE)
        .init();

    let args: Vec<String> = std::env::args().collect();
```

- [ ] **Step 2: Wrap each subcommand in a span**

Replace the `match args.get(1)...` block:
```rust
    match args.get(1).map(String::as_str) {
        Some("train") => cmd_train(&args[2..]),
        Some("selfplay") => cmd_selfplay(&args[2..]),
        Some("eval") => cmd_eval(&args[2..]),
        _ => cmd_board(&args),
    }
```
with:
```rust
    match args.get(1).map(String::as_str) {
        Some("train") => {
            let _span = info_span!("train").entered();
            cmd_train(&args[2..]);
        }
        Some("selfplay") => {
            let _span = info_span!("selfplay").entered();
            cmd_selfplay(&args[2..]);
        }
        Some("eval") => {
            let _span = info_span!("position-eval").entered();
            cmd_eval(&args[2..]);
        }
        _ => cmd_board(&args),
    }
```

- [ ] **Step 3: Verify subscriber initializes without panic**

Run the default board display command and confirm no panic occurs:
```bash
RUST_LOG=info cargo run -p cli 2>&1 | head -5
```
Expected: the ASCII board is printed; no tracing noise for a command that emits no events.

Run selfplay (which will emit debug events once Task 4 is done):
```bash
RUST_LOG=info cargo run -p cli -- selfplay --games 1 2>&1 | head -10
```
Expected: no panic; span close event visible in stderr.

- [ ] **Step 4: Commit**

```bash
git add crates/cli/src/main.rs
git commit -m "feat: initialize tracing subscriber in CLI"
```

---

### Task 3: Instrument training loop

**Files:**
- Modify: `crates/engine/src/train.rs`

- [ ] **Step 1: Add tracing imports**

In `crates/engine/src/train.rs`, add to the existing `use` block:
```rust
use tracing::{debug, info, info_span};
```

- [ ] **Step 2: Replace training start println! with info!**

Replace lines 56–61 (the println! about "Training on N positions"):
```rust
    println!(
        "Training on {} positions for {} epochs (batch={})",
        dataset.len(),
        cfg.epochs,
        cfg.batch_size
    );
```
with:
```rust
    info!(
        positions = dataset.len(),
        epochs = cfg.epochs,
        batch_size = cfg.batch_size,
        "starting training"
    );
```

- [ ] **Step 3: Add epoch span and replace epoch println! with info!**

Replace the epoch loop opening:
```rust
    for epoch in 0..cfg.epochs {
        dataset.shuffle(epoch as u64);
        let mut total_loss = 0.0f32;
        let mut n_batches = 0usize;

        for batch in dataset.batches(cfg.batch_size) {
```
with:
```rust
    for epoch in 0..cfg.epochs {
        let epoch_span = info_span!("epoch", n = epoch + 1, total = cfg.epochs);
        let _epoch_guard = epoch_span.enter();

        dataset.shuffle(epoch as u64);
        let mut total_loss = 0.0f32;
        let mut n_batches = 0usize;

        for batch in dataset.batches(cfg.batch_size) {
```

Add a `debug!` event inside the batch loop, right after `n_batches += 1;`:
```rust
            total_loss += loss.data()[0];
            n_batches += 1;
            debug!(batch = n_batches, loss = loss.data()[0], "batch");
```

Replace the epoch summary `println!` (last line of the epoch loop):
```rust
        println!("Epoch {:>3}: avg_loss = {avg:.6}", epoch + 1);
```
with:
```rust
        info!(avg_loss = avg, "epoch complete");
```

- [ ] **Step 4: Verify no compiler warnings**

```bash
cargo check -p engine
```
Expected: no errors or warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/train.rs
git commit -m "feat: add tracing spans and events to training loop"
```

---

### Task 4: Instrument selfplay

**Files:**
- Modify: `crates/engine/src/selfplay.rs`

- [ ] **Step 1: Add tracing imports**

In `crates/engine/src/selfplay.rs`, add to the existing `use` block:
```rust
use tracing::{debug, info_span};
```

- [ ] **Step 2: Replace progress println! and add span**

Replace the entire `generate` function body:
```rust
pub fn generate(model: &HybridValueNet, num_games: usize) -> ChessDataset {
    let mut dataset = ChessDataset::new();
    for game_idx in 0..num_games {
        if (game_idx + 1) % 10 == 0 {
            println!("Self-play: {}/{} games", game_idx + 1, num_games);
        }
        let samples = play_game(model);
        dataset.extend(samples);
    }
    dataset
}
```
with:
```rust
pub fn generate(model: &HybridValueNet, num_games: usize) -> ChessDataset {
    let _span = info_span!("selfplay", total_games = num_games).entered();
    let mut dataset = ChessDataset::new();
    for game_idx in 0..num_games {
        let samples = play_game(model);
        let positions = samples.len();
        dataset.extend(samples);
        debug!(game = game_idx + 1, total = num_games, positions, "game complete");
    }
    dataset
}
```

- [ ] **Step 3: Verify**

```bash
cargo check -p engine
```
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/engine/src/selfplay.rs
git commit -m "feat: add tracing span and events to selfplay"
```

---

### Task 5: Instrument dataset loading

**Files:**
- Modify: `crates/engine/src/dataset.rs`

- [ ] **Step 1: Add tracing imports**

In `crates/engine/src/dataset.rs`, add:
```rust
use tracing::{info, warn};
```

- [ ] **Step 2: Replace println!/eprintln! in load_one**

Replace the `load_one` method:
```rust
    fn load_one(&mut self, path: &Path) {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let before = self.samples.len();
                let new_samples = if is_fen_extension(path) {
                    parse_fen_file(&text)
                } else {
                    parse_pgn(&text)
                };
                self.samples.extend(new_samples);
                println!(
                    "  Loaded {} positions from {}",
                    self.samples.len() - before,
                    path.display()
                );
            }
            Err(e) => eprintln!("Warning: cannot read {}: {e}", path.display()),
        }
    }
```
with:
```rust
    fn load_one(&mut self, path: &Path) {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let before = self.samples.len();
                let new_samples = if is_fen_extension(path) {
                    parse_fen_file(&text)
                } else {
                    parse_pgn(&text)
                };
                self.samples.extend(new_samples);
                info!(
                    samples = self.samples.len() - before,
                    path = %path.display(),
                    "loaded file"
                );
            }
            Err(e) => warn!(path = %path.display(), error = %e, "cannot read file"),
        }
    }
```

- [ ] **Step 3: Replace eprintln! in from_pgn_files directory scan**

In `from_pgn_files`, replace:
```rust
                    Err(e) => {
                        eprintln!("Warning: cannot read dir {}: {e}", path.display());
                        continue;
                    }
```
with:
```rust
                    Err(e) => {
                        warn!(path = %path.display(), error = %e, "cannot read directory");
                        continue;
                    }
```

- [ ] **Step 4: Verify**

```bash
cargo check -p engine
```
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/dataset.rs
git commit -m "feat: replace println!/eprintln! with tracing events in dataset"
```

---

### Task 6: Add parse warnings in pgn.rs and fen_file.rs

**Files:**
- Modify: `crates/engine/src/pgn.rs`
- Modify: `crates/engine/src/fen_file.rs`

- [ ] **Step 1: Add warn! for skipped SAN tokens in pgn.rs**

In `crates/engine/src/pgn.rs`, in the `parse_game` function, replace the token loop:
```rust
    for token in &tokens {
        if is_result_token(token) {
            break;
        }
        if let Some(mv) = san_to_move(&board, token) {
            samples.push((board.clone(), outcome));
            board = board.make_move(mv);
        }
    }
```
with:
```rust
    for token in &tokens {
        if is_result_token(token) {
            break;
        }
        if let Some(mv) = san_to_move(&board, token) {
            samples.push((board.clone(), outcome));
            board = board.make_move(mv);
        } else {
            tracing::warn!(token, "skipped unparseable SAN token");
        }
    }
```

- [ ] **Step 2: Read fen_file.rs and add warn! for bad FEN lines**

Read `crates/engine/src/fen_file.rs` in full. Find the per-line `chess::fen::from_fen` call where parse errors are silently skipped. Add:
```rust
Err(e) => {
    tracing::warn!(line, error = %e, "skipped invalid FEN line");
}
```
at the error branch of the per-line `from_fen` call.

- [ ] **Step 3: Verify**

```bash
cargo check -p engine
```
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/engine/src/pgn.rs crates/engine/src/fen_file.rs
git commit -m "feat: add tracing warn for skipped PGN tokens and invalid FEN lines"
```

---

### Task 7: Add perft debug span and model forward trace span

**Files:**
- Modify: `crates/chess/src/movegen.rs`
- Modify: `crates/engine/src/model.rs`

- [ ] **Step 1: Add debug_span! to perft in movegen.rs**

In `crates/chess/src/movegen.rs`, replace:
```rust
pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
```
with:
```rust
pub fn perft(board: &Board, depth: u32) -> u64 {
    let _span = tracing::debug_span!("perft", depth).entered();
    if depth == 0 {
        return 1;
    }
```

- [ ] **Step 2: Add trace_span! to HybridValueNet::forward in model.rs**

In `crates/engine/src/model.rs`, add to the `use` block:
```rust
use tracing::trace_span;
```

Replace the `forward` function opening:
```rust
    pub fn forward(&self, board: &Board) -> Tensor {
        // 1. Encode → [1, 17, 8, 8]
        let x = encode_batch(board);
```
with:
```rust
    pub fn forward(&self, board: &Board) -> Tensor {
        let _span = trace_span!("HybridValueNet::forward").entered();
        // 1. Encode → [1, 17, 8, 8]
        let x = encode_batch(board);
```

- [ ] **Step 3: Verify tests still pass**

```bash
cargo test -p chess movegen
cargo test -p engine
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/chess/src/movegen.rs crates/engine/src/model.rs
git commit -m "feat: add debug_span to perft and trace_span to model forward"
```

---

### Task 8: Chess crate — piece, square, moves, game, bitboard docs

**Files:**
- Modify: `crates/chess/src/piece.rs`
- Modify: `crates/chess/src/square.rs`
- Modify: `crates/chess/src/moves.rs`
- Modify: `crates/chess/src/game.rs`
- Modify: `crates/chess/src/bitboard.rs`

- [ ] **Step 1: Document piece.rs**

Read `crates/chess/src/piece.rs`, then add `///` doc comments to every public item. Use this pattern:

```rust
/// The color of a chess piece.
pub enum Color {
    /// The side that moves first; pieces on ranks 1–2 in the starting position.
    White = 0,
    /// The side that moves second; pieces on ranks 7–8 in the starting position.
    Black = 1,
}

impl Color {
    /// Returns the opposite color.
    ///
    /// ```
    /// # use chess::piece::Color;
    /// assert_eq!(Color::White.opposite(), Color::Black);
    /// ```
    pub const fn opposite(self) -> Self { ... }
}

/// The type of a chess piece, independent of color.
pub enum PieceKind {
    /// Pawn — moves forward one square (two from the starting rank), captures diagonally.
    Pawn,
    /// Knight — moves in an L-shape; the only piece that jumps over others.
    Knight,
    /// Bishop — slides diagonally any number of squares.
    Bishop,
    /// Rook — slides along ranks and files any number of squares.
    Rook,
    /// Queen — combines rook and bishop movement.
    Queen,
    /// King — moves one square in any direction; may castle.
    King,
}

impl PieceKind {
    /// All six piece kinds in the order used by `Board::pieces` (matches `index()`).
    pub const ALL: [Self; 6] = [...];

    /// Returns the index into `Board::pieces[color]` for this piece kind (0–5).
    pub const fn index(self) -> usize { ... }

    /// Returns the uppercase FEN character for this piece kind (e.g. `'K'` for King).
    pub const fn fen_char(self) -> char { ... }
}

/// A chess piece with a specific kind and color.
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
}
```

- [ ] **Step 2: Document square.rs**

Read `crates/chess/src/square.rs`, then add `///` to all public items:

```rust
/// A board square identified by a 0–63 index (a1 = 0, h8 = 63).
///
/// Index layout: file increases left-to-right (a=0, h=7),
/// rank increases bottom-to-top (1=0, 8=7). So square index = rank * 8 + file.
pub struct Square(u8);

impl Square {
    /// Creates a `Square` from a raw 0–63 index. No bounds checking is performed.
    pub const fn from_index(index: u8) -> Self { ... }

    /// Creates a `Square` from file (0–7 mapping to a–h) and rank (0–7 mapping to 1–8).
    pub const fn from_file_rank(file: u8, rank: u8) -> Self { ... }

    /// Parses algebraic notation such as `"e4"`. Returns `None` if the string is invalid.
    pub fn from_algebraic(s: &str) -> Option<Self> { ... }

    /// Returns the 0–63 index.
    pub const fn index(self) -> u8 { ... }

    /// Returns the file index (0 = a-file, 7 = h-file).
    pub const fn file(self) -> u8 { ... }

    /// Returns the rank index (0 = rank 1, 7 = rank 8).
    pub const fn rank(self) -> u8 { ... }

    /// Returns a single-bit `u64` mask with this square's bit set (bit `index`).
    pub const fn bit(self) -> u64 { ... }
}
```

- [ ] **Step 3: Document moves.rs**

Read `crates/chess/src/moves.rs`, then add `///`:

```rust
/// The kind of a chess move.
pub enum MoveKind {
    /// A standard move or capture with no special rules.
    Normal,
    /// King moves two squares toward a rook; the rook jumps to the other side of the king.
    Castling,
    /// Pawn captures diagonally onto an empty square, removing the adjacent enemy pawn.
    EnPassant,
}

/// A chess move from one square to another, with optional promotion piece.
pub struct Move {
    /// The square the piece moves from.
    pub from: Square,
    /// The square the piece moves to.
    pub to: Square,
    /// The piece kind to promote to. `Some` only for pawn promotion moves.
    pub promotion: Option<PieceKind>,
    /// The type of this move.
    pub kind: MoveKind,
}

impl Move {
    /// Creates a normal (non-special) move.
    pub const fn normal(from: Square, to: Square) -> Self { ... }

    /// Creates a castling move. `to` is the king's destination square.
    pub const fn castling(from: Square, to: Square) -> Self { ... }

    /// Creates an en passant capture move.
    pub const fn en_passant(from: Square, to: Square) -> Self { ... }

    /// Creates a pawn promotion move.
    pub const fn promotion(from: Square, to: Square, kind: PieceKind) -> Self { ... }
}
```

- [ ] **Step 4: Document game.rs**

Read `crates/chess/src/game.rs`, then add `///`:

```rust
/// The reason a game ended in a draw.
pub enum DrawReason {
    /// Neither side has made a capture or pawn move in the last 50 full moves.
    FiftyMoveRule,
    /// Neither side has enough remaining material to force checkmate.
    InsufficientMaterial,
}

/// The current status of a chess game.
pub enum GameStatus {
    /// The game is still in progress; the side to move has at least one legal move.
    Ongoing,
    /// The side to move has no legal moves and their king is in check.
    Checkmate,
    /// The side to move has no legal moves and their king is not in check.
    Stalemate,
    /// The game ended in a draw for the given reason.
    Draw(DrawReason),
}

/// Returns the current status of the game.
///
/// Checks for checkmate, stalemate, fifty-move rule, and insufficient material.
///
/// Note: threefold repetition is not detected because `Board` stores no history.
pub fn game_status(board: &Board) -> GameStatus { ... }
```

- [ ] **Step 5: Document bitboard.rs**

Read `crates/chess/src/bitboard.rs`, then add `///`:

```rust
/// A 64-bit integer where each bit represents a board square.
///
/// Bit 0 = a1, bit 7 = h1, bit 8 = a2, …, bit 63 = h8.
pub type Bitboard = u64;

/// Sets the bit for `sq` in `bb`.
pub fn set_bit(bb: &mut u64, sq: Square) { ... }

/// Clears the bit for `sq` in `bb`.
pub fn clear_bit(bb: &mut u64, sq: Square) { ... }

/// Returns the square of the least-significant set bit without modifying `bb`.
///
/// # Panics
/// Panics if `bb` is 0.
pub fn lsb_square(bb: u64) -> Square { ... }

/// Removes and returns the least-significant set bit as a `Square`, modifying `bb` in place.
///
/// # Panics
/// Panics if `bb` is 0.
pub fn pop_lsb(bb: &mut u64) -> Square { ... }

/// Returns the number of set bits (popcount).
pub fn count_bits(bb: u64) -> u32 { ... }
```

- [ ] **Step 6: Verify doc tests compile**

```bash
cargo test --doc -p chess
```
Expected: all doc tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/chess/src/piece.rs crates/chess/src/square.rs crates/chess/src/moves.rs crates/chess/src/game.rs crates/chess/src/bitboard.rs
git commit -m "docs: add public API documentation to chess piece/square/moves/game/bitboard"
```

---

### Task 9: Chess crate — board.rs and fen.rs docs

**Files:**
- Modify: `crates/chess/src/board.rs`
- Modify: `crates/chess/src/fen.rs`

- [ ] **Step 1: Document board.rs**

Read `crates/chess/src/board.rs`, then add `///` to all public items:

```rust
/// The complete game state for a chess position.
///
/// Stores piece locations as a `[color][kind]` array of bitboards.
/// Index 0 = White, 1 = Black. Piece kind indices follow `PieceKind::index()`.
///
/// `Board` is `Clone` — `make_move` returns a new `Board` rather than mutating.
pub struct Board {
    /// Bitboard array: `pieces[color_index][kind_index]`.
    pub pieces: [[u64; 6]; 2],
    /// The side that is next to move.
    pub side_to_move: Color,
    /// Castling availability bitmask: bit 0=WK, 1=WQ, 2=BK, 3=BQ.
    pub castling: u8,
    /// En passant target square, set when the last move was a double pawn push.
    pub en_passant: Option<Square>,
    /// Half-move clock for the fifty-move draw rule.
    pub halfmove_clock: u8,
    /// Full move counter, incremented after Black's move.
    pub fullmove_number: u16,
}

impl Board {
    /// Returns the standard chess starting position.
    pub fn starting_position() -> Self { ... }

    /// Returns a bitboard of all White-occupied squares.
    pub fn white_occupied(&self) -> u64 { ... }

    /// Returns a bitboard of all Black-occupied squares.
    pub fn black_occupied(&self) -> u64 { ... }

    /// Returns a bitboard of all occupied squares (White ∪ Black).
    pub fn all_occupied(&self) -> u64 { ... }

    /// Returns the piece on `sq`, or `None` if the square is empty.
    pub fn piece_at(&self, sq: Square) -> Option<Piece> { ... }

    /// Returns `true` if `color`'s king is currently in check.
    pub fn is_in_check(&self, color: Color) -> bool { ... }

    /// Applies `mv` and returns the resulting board state.
    ///
    /// Does not validate move legality — call `generate_legal_moves` first if needed.
    pub fn make_move(&self, mv: Move) -> Self { ... }

    /// Encodes this position as a FEN string.
    pub fn to_fen(&self) -> String { ... }
}
```

- [ ] **Step 2: Document FenError variants in fen.rs**

In `crates/chess/src/fen.rs`, add `///` to each `FenError` variant:

```rust
/// Errors that can occur while parsing a FEN string.
#[derive(Debug)]
pub enum FenError {
    /// The FEN string does not have exactly 6 space-separated fields.
    WrongFieldCount(usize),
    /// A character in the piece placement field is not a valid piece letter.
    InvalidPieceChar(char),
    /// A square string (e.g. the en passant target) could not be parsed.
    InvalidSquare(String),
    /// The castling rights field contains an unrecognised character.
    InvalidCastling(String),
    /// The side-to-move field is not `"w"` or `"b"`.
    InvalidSideToMove(char),
    /// A numeric field (halfmove clock or fullmove number) failed integer parsing.
    ParseInt(ParseIntError),
}
```

- [ ] **Step 3: Verify**

```bash
cargo test --doc -p chess && cargo check -p chess
```
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/chess/src/board.rs crates/chess/src/fen.rs
git commit -m "docs: add public API documentation to chess board and FenError"
```

---

### Task 10: Chess crate — attack.rs and movegen.rs architecture docs

**Files:**
- Modify: `crates/chess/src/attack.rs`
- Modify: `crates/chess/src/movegen.rs`

- [ ] **Step 1: Add //! architecture header to attack.rs**

Read `crates/chess/src/attack.rs` in full, then add this `//!` module header at the very top (before any `use` statements):

```rust
//! Pre-computed attack tables and ray-casting for all piece types.
//!
//! ## Tables
//!
//! Attack tables are initialised once at first use via [`std::sync::OnceLock`] and
//! stored in static memory. There are four table types:
//!
//! - **Knight** — 64 precomputed bitboards, one per square.
//! - **King** — 64 precomputed bitboards, one per square.
//! - **Pawn** — 2 × 64 precomputed bitboards (one set per color).
//! - **Ray** — 64 × 8 precomputed bitboards. Each entry is the set of squares
//!   reachable from a given square in one of 8 directions, excluding the origin.
//!
//! ## Sliding piece attacks (ray casting)
//!
//! The 8 directions are indexed: N=0, NE=1, E=2, NW=3, S=4, SW=5, W=6, SE=7.
//!
//! For each ray direction from a given square:
//! 1. Intersect the precomputed ray bitboard with the occupied squares.
//! 2. Find the first blocker along the ray (LSB for positive rays, MSB for negative).
//! 3. Mask out everything beyond that blocker — the blocker's own square is included
//!    (it can be captured).
//!
//! Using precomputed rays makes each direction O(1). Rook attacks = N + S + E + W rays;
//! bishop = NE + NW + SE + SW.
```

- [ ] **Step 2: Add /// to public attack functions**

```rust
/// Returns a bitboard of all squares a knight on `sq` can jump to.
pub fn knight_attacks(sq: Square) -> u64 { ... }

/// Returns a bitboard of all squares a king on `sq` can move to (does not filter check).
pub fn king_attacks(sq: Square) -> u64 { ... }

/// Returns a bitboard of the diagonal capture squares for a pawn of `color` on `sq`.
///
/// Does not include the forward push square — only the two diagonal attack squares.
pub fn pawn_attacks(color: Color, sq: Square) -> u64 { ... }

/// Returns all squares a rook on `sq` can reach, given the `occupied` bitboard.
///
/// Slides along ranks and files, stopping at (and including) the first occupied square.
pub fn rook_attacks(sq: Square, occupied: u64) -> u64 { ... }

/// Returns all squares a bishop on `sq` can reach, given the `occupied` bitboard.
pub fn bishop_attacks(sq: Square, occupied: u64) -> u64 { ... }

/// Returns all squares a queen on `sq` can reach (union of rook and bishop attacks).
pub fn queen_attacks(sq: Square, occupied: u64) -> u64 { ... }

/// Returns the union of all attack squares reachable by any piece of `color`.
///
/// Used for check detection and castling legality — squares attacked by the enemy
/// cannot be crossed or occupied by a castling king.
pub fn all_attacks(color: Color, pieces: &[[u64; 6]; 2], occupied: u64) -> u64 { ... }
```

- [ ] **Step 3: Add //! architecture header to movegen.rs**

Add this `//!` module header at the top of `crates/chess/src/movegen.rs`:

```rust
//! Legal move generation via pseudo-legal filtering.
//!
//! ## Two-pass approach
//!
//! 1. **Pseudo-legal generation** — produces all moves that follow each piece's
//!    movement rules, without checking whether the king is left in check.
//!    This is fast and simple to implement.
//!
//! 2. **Legality filter** — for each pseudo-legal move, `make_move` is applied to a
//!    temporary board and `is_in_check` tests whether the moving side's king is
//!    attacked. Moves that leave the king in check are discarded.
//!
//! This is correct by construction: every move that passes the filter is legal.
//! The cost is one `make_move` + `is_in_check` per candidate (typically 20–80).
//!
//! ## Perft
//!
//! [`perft`] counts leaf nodes at a given depth and is the standard method for
//! validating move generators. `perft(start, 3)` must return `8902`.
```

- [ ] **Step 4: Verify**

```bash
cargo doc -p chess --no-deps 2>&1 | grep -i warning | head -20
cargo test -p chess
```
Expected: no doc warnings, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/chess/src/attack.rs crates/chess/src/movegen.rs
git commit -m "docs: add architecture docs and API docs to chess attack and movegen"
```

---

### Task 11: Tensor crate — ops.rs documentation

**Files:**
- Modify: `crates/tensor/src/ops.rs`

- [ ] **Step 1: Read ops.rs to understand all function signatures**

Read `crates/tensor/src/ops.rs` in full before writing any docs.

- [ ] **Step 2: Document element-wise and scalar ops**

Add `///` to each op. Pattern (apply consistently to `add`, `sub`, `mul`, `div`, `abs`, `pow`):

```rust
/// Adds two tensors element-wise. Both tensors must have identical shapes.
///
/// Supports autograd: gradients flow to both inputs.
pub fn add(a: &Tensor, b: &Tensor) -> Tensor { ... }

/// Subtracts `b` from `a` element-wise. Both tensors must have identical shapes.
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor { ... }

/// Multiplies all elements by a scalar constant.
///
/// Gradient: upstream × scalar.
pub fn mul_scalar(t: &Tensor, scalar: f32) -> Tensor { ... }

/// Multiplies two tensors element-wise. Both tensors must have identical shapes.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor { ... }

/// Divides `a` by `b` element-wise. Both tensors must have identical shapes.
pub fn div(a: &Tensor, b: &Tensor) -> Tensor { ... }

/// Returns the absolute value of each element.
pub fn abs(t: &Tensor) -> Tensor { ... }

/// Raises each element to the power `exp`.
pub fn pow(t: &Tensor, exp: f32) -> Tensor { ... }
```

- [ ] **Step 3: Document activation functions**

Apply to `relu`, `sigmoid`, `tanh`, `gelu`, `silu`:

```rust
/// Applies the rectified linear unit: `max(0, x)` element-wise.
pub fn relu(t: &Tensor) -> Tensor { ... }

/// Applies the sigmoid function element-wise, mapping ℝ → (0, 1).
pub fn sigmoid(t: &Tensor) -> Tensor { ... }

/// Applies the hyperbolic tangent element-wise, mapping ℝ → (-1, 1).
pub fn tanh(t: &Tensor) -> Tensor { ... }

/// Applies the Gaussian Error Linear Unit activation element-wise.
///
/// Approximation: `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`.
pub fn gelu(t: &Tensor) -> Tensor { ... }

/// Applies the Sigmoid Linear Unit: `x · sigmoid(x)` element-wise.
pub fn silu(t: &Tensor) -> Tensor { ... }
```

- [ ] **Step 4: Document aggregation and shape ops**

Apply to `mean`, `sum`, `sqrt`, `reshape`, `permute`, `cat`, `select_row`, `select_col`, `transpose`:

```rust
/// Returns the mean of all elements as a scalar tensor of shape `[1]`.
pub fn mean(t: &Tensor) -> Tensor { ... }

/// Returns the sum of all elements as a scalar tensor of shape `[1]`.
pub fn sum(t: &Tensor) -> Tensor { ... }

/// Returns the element-wise square root.
pub fn sqrt(t: &Tensor) -> Tensor { ... }

/// Returns a view of `t` with a new shape. Total element count must be unchanged.
pub fn reshape(t: &Tensor, shape: &[usize]) -> Tensor { ... }

/// Reorders the dimensions of `t` according to `axes`.
///
/// `axes` must be a permutation of `0..t.shape().len()`.
pub fn permute(t: &Tensor, axes: &[usize]) -> Tensor { ... }

/// Concatenates tensors along the first (row) dimension.
///
/// All tensors must have the same shape in every dimension except the first.
pub fn cat(tensors: &[&Tensor]) -> Tensor { ... }

/// Extracts a single row from a 2-D tensor as a 1-D tensor.
///
/// `t` must have shape `[rows, cols]`; result has shape `[cols]`.
pub fn select_row(t: &Tensor, row: usize) -> Tensor { ... }

/// Extracts a single column from a 2-D tensor as a 1-D tensor.
///
/// `t` must have shape `[rows, cols]`; result has shape `[rows]`.
pub fn select_col(t: &Tensor, col: usize) -> Tensor { ... }

/// Transposes a 2-D tensor, swapping rows and columns.
///
/// Input shape `[m, n]` → output shape `[n, m]`.
pub fn transpose(t: &Tensor) -> Tensor { ... }
```

- [ ] **Step 5: Document matrix, pooling, norm, and loss ops**

Apply to `matmul`, `max_pool2d`, `avg_pool2d`, `batch_norm`, `layer_norm`, `dropout`, `softmax`, `log_softmax`, `cross_entropy`:

```rust
/// Matrix multiplication of two 2-D tensors.
///
/// `a` must have shape `[m, k]`, `b` must have shape `[k, n]`. Result: `[m, n]`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor { ... }

/// 2-D max pooling with the given `kernel_size`. Reduces spatial dimensions by `kernel_size`.
///
/// Input: `[batch, channels, height, width]`. Kernel is square; stride equals kernel_size.
pub fn max_pool2d(t: &Tensor, kernel_size: usize) -> Tensor { ... }

/// 2-D average pooling with the given `kernel_size`.
pub fn avg_pool2d(t: &Tensor, kernel_size: usize) -> Tensor { ... }

/// Applies batch normalisation over a 4-D input `[batch, channels, height, width]`.
pub fn batch_norm(t: &Tensor, gamma: &Tensor, beta: &Tensor) -> Tensor { ... }

/// Applies layer normalisation over the last dimension.
pub fn layer_norm(t: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor { ... }

/// Applies dropout with probability `p` during training. Pass `training = false` for inference.
pub fn dropout(t: &Tensor, p: f32, training: bool) -> Tensor { ... }

/// Applies softmax along the last dimension, normalising values to sum to 1.
///
/// Input and output have the same shape.
pub fn softmax(t: &Tensor) -> Tensor { ... }

/// Applies log-softmax along the last dimension (numerically more stable than `log(softmax(x))`).
pub fn log_softmax(t: &Tensor) -> Tensor { ... }

/// Computes cross-entropy loss between logits and integer class targets.
///
/// `logits` has shape `[batch, classes]`; `targets` is a flat `[batch]` integer tensor.
/// Returns a scalar loss tensor.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor { ... }
```

- [ ] **Step 6: Verify**

```bash
cargo doc -p tensor --no-deps 2>&1 | grep -i warning | head -20
```
Expected: no missing-docs warnings.

- [ ] **Step 7: Commit**

```bash
git add crates/tensor/src/ops.rs
git commit -m "docs: add shape contracts and descriptions to all tensor ops"
```

---

### Task 12: Tensor crate — tensor_impl.rs, nn/ layers, and optim.rs

**Files:**
- Modify: `crates/tensor/src/tensor_impl.rs`
- Modify: all files under `crates/tensor/src/nn/`
- Modify: `crates/tensor/src/optim.rs`

- [ ] **Step 1: Read all tensor_impl.rs, nn/ files, and optim.rs**

Read each file before adding docs:
- `crates/tensor/src/tensor_impl.rs`
- All files under `crates/tensor/src/nn/`
- `crates/tensor/src/optim.rs`

- [ ] **Step 2: Document Tensor struct and methods in tensor_impl.rs**

```rust
/// A multi-dimensional float32 array with optional automatic differentiation.
///
/// Tensors are reference-counted (`Arc`-backed) so cloning is cheap — all clones
/// share the same underlying data and gradient buffer.
///
/// ## Autograd
///
/// When `requires_grad` is enabled, every operation that produces a new tensor
/// records a `GradFn`. Calling `backward()` on a scalar loss tensor performs a
/// reverse-mode pass, accumulating gradients into all leaf tensors.
pub struct Tensor { ... }

impl Tensor {
    /// Creates a tensor from a `Vec<f32>` with the given shape.
    ///
    /// The product of all `shape` dimensions must equal `data.len()`.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self { ... }

    /// Creates a tensor of all zeros with the given shape.
    pub fn zeros(shape: &[usize]) -> Self { ... }

    /// Creates a tensor of all ones with the given shape.
    pub fn ones(shape: &[usize]) -> Self { ... }

    /// Creates a tensor where every element equals `val`.
    pub fn full(shape: &[usize], val: f32) -> Self { ... }

    /// Creates a tensor of independent samples from N(0, std²).
    pub fn randn(shape: &[usize], std: f32) -> Self { ... }

    /// Returns a read-only view of the underlying flat data.
    pub fn data(&self) -> impl Deref<Target = [f32]> + '_ { ... }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] { ... }

    /// Returns a tensor with the same data but a new shape. Total elements must be unchanged.
    pub fn reshape(&self, shape: &[usize]) -> Self { ... }

    /// Returns `true` if this tensor tracks gradients.
    pub fn requires_grad(&self) -> bool { ... }

    /// Enables gradient tracking on this tensor, returning `self`.
    pub fn with_grad(self) -> Self { ... }

    /// Triggers the backward pass from this scalar tensor.
    ///
    /// Propagates gradients through the computation graph to all leaf tensors.
    ///
    /// # Panics
    /// Panics if this tensor is not a scalar (shape `[1]`).
    pub fn backward(&self) { ... }

    /// Adds `grad` to this tensor's gradient accumulator.
    ///
    /// Called automatically during `backward()`; can also be used manually.
    pub fn accumulate_grad(&self, grad: &[f32]) { ... }

    /// Returns `true` if this tensor has no `grad_fn` (i.e. it is a leaf parameter).
    pub fn is_leaf(&self) -> bool { ... }
}
```

- [ ] **Step 3: Document nn/ layers**

For each layer (`Linear`, `Conv2d`, `BatchNorm2d`, `LayerNorm`, `MultiHeadAttention`, `TransformerBlock`, `TransformerEncoder`), add struct-level and method-level `///`. Use these patterns:

```rust
/// A fully-connected linear layer applying `y = x Wᵀ + b`.
///
/// Weights are shape `[out_features, in_features]`, initialised with `randn(std=0.02)`.
pub struct Linear { ... }

impl Linear {
    /// Creates a new `Linear` layer. If `bias` is `true`, adds a zero-initialised bias.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self { ... }

    /// Applies the linear transformation.
    ///
    /// Input shape: `[*, in_features]`. Output shape: `[*, out_features]`.
    pub fn forward(&self, x: &Tensor) -> Tensor { ... }

    /// Returns all learnable parameters (weight and optional bias).
    pub fn parameters(&self) -> Vec<Tensor> { ... }
}

/// A 2-D convolutional layer with same-padding (output spatial size equals input).
pub struct Conv2d { ... }

/// Batch normalisation over a `[batch, channels, H, W]` input.
///
/// Normalises each channel across the batch and spatial dimensions.
/// Learnable scale (`gamma`) and shift (`beta`) parameters are per-channel.
pub struct BatchNorm2d { ... }

/// Layer normalisation applied over the last dimension.
pub struct LayerNorm { ... }

/// Scaled dot-product multi-head attention.
///
/// Splits `d_model` into `num_heads` independent heads of dimension `d_model / num_heads`,
/// computes scaled dot-product attention for each head, then projects the concatenated
/// result back to `d_model`.
pub struct MultiHeadAttention { ... }

/// A single Transformer block: multi-head self-attention followed by a feed-forward network.
///
/// Both sub-layers use residual connections and layer normalisation (pre-norm style).
pub struct TransformerBlock { ... }

/// A stack of [`TransformerBlock`]s.
pub struct TransformerEncoder { ... }
```

- [ ] **Step 4: Document Adam in optim.rs**

```rust
/// Adam optimizer (Adaptive Moment Estimation).
///
/// Maintains per-parameter first moment (mean) and second moment (variance) estimates
/// with exponential decay rates β₁=0.9, β₂=0.999, ε=1e-8. Bias-corrects both
/// moments before applying the parameter update, which is important in early training.
///
/// # Usage
///
/// ```ignore
/// let mut adam = Adam::new(model.parameters(), 1e-4);
/// // training loop:
/// loss.backward();
/// adam.step();
/// adam.zero_grad();
/// ```
pub struct Adam { ... }

impl Adam {
    /// Creates an Adam optimizer for the given parameters with learning rate `lr`.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self { ... }

    /// Applies one gradient descent step to all managed parameters.
    pub fn step(&mut self) { ... }

    /// Zeros the gradient buffers of all managed parameters.
    ///
    /// Must be called before `backward()` each iteration to prevent gradient accumulation.
    pub fn zero_grad(&mut self) { ... }
}
```

- [ ] **Step 5: Verify**

```bash
cargo doc -p tensor --no-deps 2>&1 | grep -i warning | head -20
cargo test -p tensor
```
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/tensor/src/tensor_impl.rs crates/tensor/src/nn/ crates/tensor/src/optim.rs
git commit -m "docs: add documentation to Tensor, nn layers, and Adam optimizer"
```

---

### Task 13: Engine crate — encode.rs architecture doc + model.rs expansion

**Files:**
- Modify: `crates/engine/src/encode.rs`
- Modify: `crates/engine/src/model.rs`
- Modify: `crates/engine/src/pgn.rs`

- [ ] **Step 1: Read encode.rs**

Read `crates/engine/src/encode.rs` in full.

- [ ] **Step 2: Add //! architecture header to encode.rs**

Add at the very top of `crates/engine/src/encode.rs`:

```rust
//! Board-to-tensor encoding for the neural network.
//!
//! ## 17-Channel Plane Layout
//!
//! A board position is encoded as a `[17, 8, 8]` float32 tensor.
//! Each of the 17 channels is a binary 8×8 plane:
//!
//! | Channel | Content |
//! |---------|---------|
//! | 0–5     | White pieces: Pawn, Knight, Bishop, Rook, Queen, King |
//! | 6–11    | Black pieces: Pawn, Knight, Bishop, Rook, Queen, King |
//! | 12      | Side to move (all 1.0 = White to move, all 0.0 = Black) |
//! | 13      | White kingside castling right |
//! | 14      | White queenside castling right |
//! | 15      | Black kingside castling right |
//! | 16      | Black queenside castling right |
//!
//! ## Design rationale
//!
//! - **Piece planes (0–11):** Mirror the `Board::pieces` bitboard layout. Each
//!   square is 1.0 if that piece occupies it, 0.0 otherwise.
//! - **Side-to-move plane (12):** Gives the network an explicit global signal
//!   about whose turn it is without requiring inference from the piece positions.
//! - **Castling planes (13–16):** Constant planes (all 1.0 or all 0.0) that
//!   communicate castling availability as a simple spatial mask.
//!
//! This encoding follows the AlphaZero convention and is designed for 2-D
//! convolutional layers that treat the 8×8 board as a spatial grid.
```

- [ ] **Step 3: Add /// to encode and encode_batch**

```rust
/// Encodes a single board position into a `[17, 8, 8]` float32 tensor.
///
/// See the module-level documentation for the full channel layout.
pub fn encode(board: &Board) -> Tensor { ... }

/// Encodes a board position as a batch tensor of shape `[1, 17, 8, 8]`.
///
/// Equivalent to `encode(board).reshape(&[1, 17, 8, 8])`. This is the format
/// expected by `HybridValueNet::forward`.
pub fn encode_batch(board: &Board) -> Tensor { ... }
```

- [ ] **Step 4: Expand model.rs //! header**

In `crates/engine/src/model.rs`, append to the existing `//!` header (after the data-flow diagram):

```rust
//!
//! ## CLS token
//!
//! The CLS ("classification") token is a learnable vector prepended to the
//! 64 square tokens before the transformer encoder. By the final layer, it has
//! attended to all 64 squares and aggregates global board context. Extracting
//! only the CLS output (row 0) gives a fixed-size 256-dim representation
//! independent of board position, suitable for the scalar head.
//!
//! ## tanh output
//!
//! The final `tanh` bounds the output to (-1, +1), matching the training labels
//! (+1.0 = White wins, -1.0 = Black wins, 0.0 = draw). Bounded output also
//! stabilises MSE loss by preventing divergence during early training.
```

- [ ] **Step 5: Document Sample in pgn.rs**

Update the `Sample` type alias doc:

```rust
/// A single training sample: a board position paired with its game outcome.
///
/// The outcome label is from White's perspective:
/// - `1.0` — White wins
/// - `-1.0` — Black wins  
/// - `0.0` — draw
///
/// This matches the output range of `HybridValueNet::forward`.
pub type Sample = (Board, f32);
```

- [ ] **Step 6: Verify**

```bash
cargo doc -p engine --no-deps 2>&1 | grep -i warning | head -20
cargo check -p engine
```
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add crates/engine/src/encode.rs crates/engine/src/model.rs crates/engine/src/pgn.rs
git commit -m "docs: add 17-channel encoding doc, expand model architecture, document Sample"
```

---

### Task 14: Final verification

**Files:** All crates

- [ ] **Step 1: Run full format check**

```bash
cargo fmt --check
```
If this fails, run `cargo fmt` and `git add -u` before continuing.

- [ ] **Step 2: Run full lint check**

```bash
cargo lint
```
Expected: zero warnings. Common tracing clippy issues to fix:
- Non-primitive fields need `%value` (Display) or `?value` (Debug) formatter in macros
- Unused `_span` variables should use a leading underscore: `let _span = ...`

- [ ] **Step 3: Run all tests**

```bash
cargo test --all
```
Expected: all tests pass.

- [ ] **Step 4: Check generated documentation**

```bash
cargo doc --all --no-deps 2>&1 | grep -E "warning|error" | head -30
```
Expected: no missing-doc warnings on public items.

- [ ] **Step 5: Smoke test structured logging**

Run the board display command (produces no tracing output):
```bash
RUST_LOG=info cargo run -p cli 2>&1 | head -5
```
Expected: ASCII board only, no tracing noise.

Run selfplay and confirm debug events appear at debug level:
```bash
RUST_LOG=debug cargo run -p cli -- selfplay --games 2 2>&1 | grep -E "DEBUG|INFO" | head -15
```
Expected: `DEBUG engine::selfplay` game-complete events and `INFO selfplay` span close with elapsed time.

- [ ] **Step 6: Final commit**

```bash
git add -u
git commit -m "docs: final verification pass — all lints and tests green"
```
