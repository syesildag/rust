# UCI Protocol — Design Spec

**Date:** 2026-05-15  
**Status:** Approved  
**Scope:** Minimal UCI binary to make the chess engine playable in standard GUIs.

---

## Goal

Add a `crates/uci` binary that speaks the Universal Chess Interface (UCI) protocol over stdin/stdout. The engine becomes usable in any UCI-compatible GUI (Arena, Cutechess, Lichess local engine, etc.) and can be benchmarked against other engines.

---

## What We Are Building

A new workspace binary crate `crates/uci` with two source files:

```
crates/uci/
  Cargo.toml
  src/
    main.rs     — UciEngine struct, stdin event loop, command dispatch
    search.rs   — greedy 1-ply move selection + UCI move string parser
```

---

## UCI Commands Implemented

| Command | Engine Response | Notes |
|---|---|---|
| `uci` | `id name ...` + `id author ...` + `uciok` | Identification handshake |
| `isready` | `readyok` | Readiness check; model loaded by this point |
| `ucinewgame` | _(no output)_ | Resets board to starting position |
| `position startpos [moves ...]` | _(no output)_ | Sets board from start + optional move list |
| `position fen <fen> [moves ...]` | _(no output)_ | Sets board from FEN + optional move list |
| `go` | `bestmove <move>` | Runs greedy 1-ply search |
| `quit` | _(process exits)_ | Clean shutdown |

All unrecognised commands are silently ignored (required by UCI spec).

---

## Components

### `UciEngine` (in `main.rs`)

```rust
struct UciEngine {
    model: HybridValueNet,
    board: Board,
}
```

- Initialised on startup by loading `model.bin`. Falls back to random weights with a `tracing::warn` if the file is absent or corrupt — identical to the training loop convention.
- Owns mutable board state across multiple `position` commands within a session.
- All UCI command handlers are methods on `UciEngine`.

### `search.rs` — Two functions

**`best_move(model: &HybridValueNet, board: &Board) -> Option<Move>`**

Greedy 1-ply evaluation:
1. Generate all legal moves with `generate_legal_moves`.
2. Build `after_boards` by applying each move.
3. Call `model.forward_batch(&after_boards)`.
4. Return the move whose resulting position has the highest value from the current player's perspective (same sign logic as `selfplay::play_game`).
5. Return `None` if no legal moves (checkmate / stalemate).

This reuses the exact same scoring logic as `selfplay.rs` for consistency.

**`parse_uci_move(board: &Board, s: &str) -> Option<Move>`**

Parses a UCI long-algebraic move string (e.g. `"e2e4"`, `"e7e8q"`) into a `Move`:
1. Split `s` into `from` (chars 0–1) and `to` (chars 2–3); parse each with `Square::from_algebraic`.
2. Extract optional promotion piece from char 4 (`q/r/b/n`).
3. Match the parsed (from, to, promotion) against `generate_legal_moves(board)` — return the first legal move with matching from/to/promotion.
4. Return `None` if the string is malformed or the move is not legal.

Matching against legal moves (rather than constructing a `Move` directly) is essential because `MoveKind` (Normal / Castling / EnPassant) must be inferred from the board state.

---

## Data Flow

```
stdin line
  → trim + split on whitespace
  → match first token:
      "uci"          → print id lines + uciok
      "isready"      → print readyok
      "ucinewgame"   → engine.board = Board::starting_position()
      "position"     → parse startpos|fen, apply move list via parse_uci_move
      "go"           → best_move() → print "bestmove <move>" (or "bestmove 0000")
      "quit"         → return / process exit
      _              → ignore
```

All stdout writes must be followed by an immediate flush — GUIs block on line-buffered output.

---

## Edge Cases

| Situation | Behaviour |
|---|---|
| `go` with no legal moves | Print `bestmove 0000` (UCI convention) |
| `position moves ...` with an illegal move string | Skip that move; apply up to the last valid move |
| Model file missing / corrupt | Warn, fall back to `HybridValueNet::default()` (random weights) |
| Unknown command | Silently ignore |

---

## Testing

One unit test in `search.rs`:
```rust
#[test]
fn parse_e2e4_from_start() {
    let board = Board::starting_position();
    let mv = parse_uci_move(&board, "e2e4");
    assert!(mv.is_some());
    // from=e2, to=e4
    let mv = mv.unwrap();
    assert_eq!(mv.from, Square::from_algebraic("e2").unwrap());
    assert_eq!(mv.to,   Square::from_algebraic("e4").unwrap());
}
```

---

## Cargo.toml (crates/uci)

```toml
[package]
name    = "uci"
version.workspace = true
edition.workspace = true

[[bin]]
name = "uci"
path = "src/main.rs"

[dependencies]
engine  = { path = "../engine" }
chess   = { path = "../chess" }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
```

Root `Cargo.toml` workspace `members` gains `"crates/uci"`.

---

## Out of Scope

- `movetime` / `depth` / `nodes` options on `go`
- `info` lines (score, depth, pv)
- `setoption`
- Pondering (`ponderhit`)
- Multi-PV

These can be added incrementally once the minimal loop works.
