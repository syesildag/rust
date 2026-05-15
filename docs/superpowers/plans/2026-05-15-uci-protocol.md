# UCI Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `crates/uci` binary that speaks the minimal UCI protocol over stdin/stdout, making the chess engine playable in any UCI-compatible GUI.

**Architecture:** New workspace binary crate with two source files. `search.rs` owns greedy move selection and UCI move string parsing; `main.rs` owns the `UciEngine` struct and the stdin event loop. The engine uses the existing `HybridValueNet` and `Board` types without modification.

**Tech Stack:** Rust stable, `engine` crate (`HybridValueNet`, `Persist` trait), `chess` crate (`Board`, `chess::fen`, `movegen`, `moves`, `piece::PieceKind`, `square::Square`), `tracing` + `tracing-subscriber` for stderr logging.

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `crates/uci/Cargo.toml` | Create | Crate manifest; depends on `engine`, `chess`, `tracing` |
| `crates/uci/src/main.rs` | Create | `UciEngine` struct + stdin event loop + command dispatch |
| `crates/uci/src/search.rs` | Create | `parse_uci_move` + `best_move` + unit tests |
| `Cargo.toml` (root) | Modify | Add `"crates/uci"` to workspace `members` |

---

## Task 1: Scaffold the crate

**Files:**
- Create: `crates/uci/Cargo.toml`
- Create: `crates/uci/src/main.rs` (stub)
- Create: `crates/uci/src/search.rs` (stub)
- Modify: `Cargo.toml` (root workspace)

- [ ] **Step 1: Create the crate manifest**

Create `crates/uci/Cargo.toml`:

```toml
[package]
name = "uci"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[lints]
workspace = true

[[bin]]
name = "uci"
path = "src/main.rs"

[dependencies]
engine = { path = "../engine" }
chess  = { path = "../chess" }
tracing            = { workspace = true }
tracing-subscriber = { workspace = true }
```

- [ ] **Step 2: Add crate to workspace**

In root `Cargo.toml`, add `"crates/uci"` to the `members` list:

```toml
members = ["crates/core", "crates/cli", "crates/chess", "crates/tensor", "crates/engine", "crates/plot", "crates/uci"]
```

- [ ] **Step 3: Create stub source files**

Create `crates/uci/src/search.rs`:

```rust
use chess::board::Board;
use chess::moves::Move;
use engine::model::HybridValueNet;

pub fn best_move(_model: &HybridValueNet, _board: &Board) -> Option<Move> {
    todo!()
}

pub fn parse_uci_move(_board: &Board, _s: &str) -> Option<Move> {
    todo!()
}
```

Create `crates/uci/src/main.rs`:

```rust
mod search;

fn main() {}
```

- [ ] **Step 4: Verify it compiles**

```bash
cargo build -p uci
```

Expected: compiles (with `todo!()` stubs, no run needed).

- [ ] **Step 5: Commit scaffold**

```bash
git add crates/uci/ Cargo.toml Cargo.lock
git commit -m "chore(uci): scaffold new crate with stub source files"
```

---

## Task 2: Implement `parse_uci_move` (TDD)

**Files:**
- Modify: `crates/uci/src/search.rs`

- [ ] **Step 1: Write failing tests**

Replace the stub `parse_uci_move` and add tests at the bottom of `crates/uci/src/search.rs`:

```rust
use chess::board::Board;
use chess::moves::Move;
use chess::piece::PieceKind;
use chess::square::Square;
use chess::movegen::generate_legal_moves;
use engine::model::HybridValueNet;

pub fn best_move(_model: &HybridValueNet, _board: &Board) -> Option<Move> {
    todo!()
}

pub fn parse_uci_move(_board: &Board, _s: &str) -> Option<Move> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_e2e4_from_start() {
        let board = Board::starting_position();
        let mv = parse_uci_move(&board, "e2e4");
        assert!(mv.is_some());
        let mv = mv.unwrap();
        assert_eq!(mv.from, Square::from_algebraic("e2").unwrap());
        assert_eq!(mv.to, Square::from_algebraic("e4").unwrap());
        assert_eq!(mv.promotion, None);
    }

    #[test]
    fn parse_promotion_move() {
        // White pawn on e7, ready to promote.
        let board = chess::fen::from_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1").unwrap();
        let mv = parse_uci_move(&board, "e7e8q");
        assert!(mv.is_some());
        let mv = mv.unwrap();
        assert_eq!(mv.promotion, Some(PieceKind::Queen));
    }

    #[test]
    fn parse_illegal_move_returns_none() {
        let board = Board::starting_position();
        assert!(parse_uci_move(&board, "e2e5").is_none()); // illegal jump
    }

    #[test]
    fn parse_malformed_returns_none() {
        let board = Board::starting_position();
        assert!(parse_uci_move(&board, "xyz").is_none());
        assert!(parse_uci_move(&board, "").is_none());
    }
}
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cargo test -p uci parse_uci_move 2>&1 | head -20
```

Expected: panics at `todo!()`.

- [ ] **Step 3: Implement `parse_uci_move`**

Replace the `parse_uci_move` stub with the real implementation:

```rust
pub fn parse_uci_move(board: &Board, s: &str) -> Option<Move> {
    if s.len() < 4 {
        return None;
    }
    let from = Square::from_algebraic(&s[0..2])?;
    let to   = Square::from_algebraic(&s[2..4])?;
    let promo = s.chars().nth(4).and_then(|c| match c {
        'q' => Some(PieceKind::Queen),
        'r' => Some(PieceKind::Rook),
        'b' => Some(PieceKind::Bishop),
        'n' => Some(PieceKind::Knight),
        _   => None,
    });
    generate_legal_moves(board)
        .into_iter()
        .find(|mv| mv.from == from && mv.to == to && mv.promotion == promo)
}
```

The full `search.rs` at this point:

```rust
use chess::board::Board;
use chess::movegen::generate_legal_moves;
use chess::moves::Move;
use chess::piece::PieceKind;
use chess::square::Square;
use engine::model::HybridValueNet;

pub fn best_move(_model: &HybridValueNet, _board: &Board) -> Option<Move> {
    todo!()
}

pub fn parse_uci_move(board: &Board, s: &str) -> Option<Move> {
    if s.len() < 4 {
        return None;
    }
    let from = Square::from_algebraic(&s[0..2])?;
    let to   = Square::from_algebraic(&s[2..4])?;
    let promo = s.chars().nth(4).and_then(|c| match c {
        'q' => Some(PieceKind::Queen),
        'r' => Some(PieceKind::Rook),
        'b' => Some(PieceKind::Bishop),
        'n' => Some(PieceKind::Knight),
        _   => None,
    });
    generate_legal_moves(board)
        .into_iter()
        .find(|mv| mv.from == from && mv.to == to && mv.promotion == promo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_e2e4_from_start() {
        let board = Board::starting_position();
        let mv = parse_uci_move(&board, "e2e4");
        assert!(mv.is_some());
        let mv = mv.unwrap();
        assert_eq!(mv.from, Square::from_algebraic("e2").unwrap());
        assert_eq!(mv.to, Square::from_algebraic("e4").unwrap());
        assert_eq!(mv.promotion, None);
    }

    #[test]
    fn parse_promotion_move() {
        let board = chess::fen::from_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1").unwrap();
        let mv = parse_uci_move(&board, "e7e8q");
        assert!(mv.is_some());
        let mv = mv.unwrap();
        assert_eq!(mv.promotion, Some(PieceKind::Queen));
    }

    #[test]
    fn parse_illegal_move_returns_none() {
        let board = Board::starting_position();
        assert!(parse_uci_move(&board, "e2e5").is_none());
    }

    #[test]
    fn parse_malformed_returns_none() {
        let board = Board::starting_position();
        assert!(parse_uci_move(&board, "xyz").is_none());
        assert!(parse_uci_move(&board, "").is_none());
    }
}
```

- [ ] **Step 4: Run tests — expect pass**

```bash
cargo test -p uci 2>&1 | tail -15
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/uci/src/search.rs
git commit -m "feat(uci): implement parse_uci_move with legal-move matching"
```

---

## Task 3: Implement `best_move` (TDD)

**Files:**
- Modify: `crates/uci/src/search.rs`

- [ ] **Step 1: Write the failing test**

Add this test to the `tests` module in `search.rs`:

```rust
#[test]
fn best_move_returns_legal_move() {
    let model = HybridValueNet::new();
    model.set_training(false);
    let board = Board::starting_position();
    let legal = generate_legal_moves(&board);
    let mv = best_move(&model, &board);
    assert!(mv.is_some());
    assert!(legal.contains(&mv.unwrap()));
}

#[test]
fn best_move_returns_none_when_no_legal_moves() {
    // Checkmate position: white is checkmated (Fool's Mate result).
    // rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3
    let board = chess::fen::from_fen(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    )
    .unwrap();
    assert!(best_move(&HybridValueNet::new(), &board).is_none());
}
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cargo test -p uci best_move 2>&1 | head -20
```

Expected: panics at `todo!()`.

- [ ] **Step 3: Implement `best_move`**

Replace the `best_move` stub:

```rust
pub fn best_move(model: &HybridValueNet, board: &Board) -> Option<Move> {
    use chess::piece::Color;
    let legal = generate_legal_moves(board);
    if legal.is_empty() {
        return None;
    }
    let after_boards: Vec<Board> = legal.iter().copied().map(|mv| board.make_move(mv)).collect();
    let raw = model.forward_batch(&after_boards).data();
    let sign = match board.side_to_move {
        Color::White =>  1.0_f32,
        Color::Black => -1.0_f32,
    };
    (0..legal.len())
        .max_by(|&i, &j| {
            (sign * raw[i])
                .partial_cmp(&(sign * raw[j]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|i| legal[i])
}
```

- [ ] **Step 4: Run tests — expect pass**

```bash
cargo test -p uci 2>&1 | tail -15
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/uci/src/search.rs
git commit -m "feat(uci): implement greedy 1-ply best_move"
```

---

## Task 4: Implement `UciEngine` and the stdin event loop

**Files:**
- Modify: `crates/uci/src/main.rs`

- [ ] **Step 1: Write `main.rs`**

Replace the stub `main.rs` entirely:

```rust
mod search;

use chess::board::Board;
use engine::model::HybridValueNet;
use engine::persist::Persist;
use search::{best_move, parse_uci_move};
use std::io::{self, BufRead, Write};
use tracing::warn;

const MODEL_PATH: &str = "model.bin";
const ENGINE_NAME: &str = "HybridNet";
const ENGINE_AUTHOR: &str = "serkan";

struct UciEngine {
    model: HybridValueNet,
    board: Board,
}

impl UciEngine {
    fn new() -> Self {
        let model = HybridValueNet::load_from(std::path::Path::new(MODEL_PATH))
            .inspect_err(|e| warn!(error = %e, "no saved model — using random weights"))
            .unwrap_or_default();
        model.set_training(false);
        Self { model, board: Board::starting_position() }
    }

    fn handle_position(&mut self, tokens: &[&str]) {
        let moves_idx = tokens.iter().position(|&t| t == "moves");
        let move_tokens = moves_idx.map_or(&[][..], |i| &tokens[i + 1..]);

        self.board = if tokens.first() == Some(&"startpos") {
            Board::starting_position()
        } else if tokens.first() == Some(&"fen") {
            let fen_end = moves_idx.unwrap_or(tokens.len());
            let fen = tokens[1..fen_end].join(" ");
            chess::fen::from_fen(&fen)
                .inspect_err(|e| warn!(error = %e, %fen, "invalid FEN — using starting position"))
                .unwrap_or_else(|_| Board::starting_position())
        } else {
            Board::starting_position()
        };

        for mv_str in move_tokens {
            if let Some(mv) = parse_uci_move(&self.board, mv_str) {
                self.board = self.board.make_move(mv);
            }
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .init();

    let mut engine = UciEngine::new();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in io::stdin().lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.as_slice() {
            ["uci"] => {
                writeln!(out, "id name {ENGINE_NAME}").ok();
                writeln!(out, "id author {ENGINE_AUTHOR}").ok();
                writeln!(out, "uciok").ok();
                out.flush().ok();
            }
            ["isready"] => {
                writeln!(out, "readyok").ok();
                out.flush().ok();
            }
            ["ucinewgame"] => {
                engine.board = Board::starting_position();
            }
            ["position", rest @ ..] => {
                engine.handle_position(rest);
            }
            ["go", ..] => {
                let mv_str = match best_move(&engine.model, &engine.board) {
                    Some(mv) => mv.to_string(),
                    None => "0000".to_string(),
                };
                writeln!(out, "bestmove {mv_str}").ok();
                out.flush().ok();
            }
            ["quit"] => break,
            _ => {}
        }
    }
}
```

- [ ] **Step 2: Build and run a smoke test**

```bash
cargo build -p uci 2>&1 | tail -5
```

Expected: `Finished` with no errors.

Then send a minimal UCI session:

```bash
printf 'uci\nisready\nposition startpos\ngo\nquit\n' | ./target/debug/uci 2>/dev/null
```

Expected output (exact format):
```
id name HybridNet
id author serkan
uciok
readyok
bestmove <some move like e2e4>
```

- [ ] **Step 3: Test a FEN position**

```bash
printf 'position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\ngo\nquit\n' \
  | ./target/debug/uci 2>/dev/null
```

Expected: `bestmove <some Black move>` (e.g. `e7e5`).

- [ ] **Step 4: Test position with moves sequence**

```bash
printf 'position startpos moves e2e4 e7e5\ngo\nquit\n' \
  | ./target/debug/uci 2>/dev/null
```

Expected: `bestmove <some White move>` (position after 1.e4 e5, White to move).

- [ ] **Step 5: Run lint**

```bash
cargo lint 2>&1 | tail -5
```

Expected: `warning: ... generated X warnings` but exit 0. Fix any errors if exit non-zero.

- [ ] **Step 6: Commit**

```bash
git add crates/uci/src/main.rs
git commit -m "feat(uci): implement UciEngine with stdin event loop and command dispatch"
```

---

## Task 5: Final check and release build

**Files:** none new

- [ ] **Step 1: Run all tests**

```bash
cargo test --all 2>&1 | tail -15
```

Expected: all tests pass (including the 6 in `crates/uci`).

- [ ] **Step 2: Run full check**

```bash
cargo check-all 2>&1 | tail -10
```

Expected: `warning: ... generated X warnings` (pedantic clippy may warn on unused imports etc.). Exit 0.

- [ ] **Step 3: Release build**

```bash
cargo build -p uci --release 2>&1 | tail -5
```

Expected: `Finished release` with the binary at `./target/release/uci`.

- [ ] **Step 4: Final smoke test with release binary**

```bash
printf 'uci\nisready\nposition startpos moves e2e4\ngo\nquit\n' \
  | ./target/release/uci 2>/dev/null
```

Expected:
```
id name HybridNet
id author serkan
uciok
readyok
bestmove <black's best response>
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(uci): add minimal UCI binary — engine playable in standard GUIs"
```

---

## Connecting to a GUI (post-implementation notes)

To use the engine in **Arena** or **Cutechess**:
1. Build with `cargo build -p uci --release`
2. In the GUI, add a new engine and point it at `./target/release/uci`
3. The engine will load `model.bin` from the **current working directory** when launched — set the GUI's engine working directory accordingly, or copy `model.bin` next to the binary.

To test against **Stockfish** via `cutechess-cli`:
```bash
cutechess-cli \
  -engine cmd=./target/release/uci \
  -engine cmd=stockfish \
  -each proto=uci tc=inf/5+0 \
  -rounds 10
```
