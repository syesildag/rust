# `uci` — HybridNet Chess Engine

A minimal [UCI (Universal Chess Interface)](https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html) binary that lets any UCI-compatible GUI (En Croissant, Arena, CuteChess, …) play games against the `HybridNet` neural network.

---

## Crate layout

```
crates/uci/
├── Cargo.toml
└── src/
    ├── main.rs   — UCI protocol loop, engine state, model loading
    └── search.rs — move selection and UCI move parsing
```

---

## Building

```bash
# debug
cargo build -p uci

# release (use this for GUIs)
cargo build -p uci --release
```

The binary is written to `target/release/uci`.

---

## Model weights

On startup the engine looks for `model.bin` in this order:

1. **Same directory as the binary** — `$(dirname $(which uci))/model.bin`
2. **Current working directory** — `./model.bin`

If neither exists the engine starts with random weights and logs a warning to stderr. Copy a trained checkpoint alongside the binary to enable learned play:

```bash
cp model.bin target/release/model.bin
```

> **Why this lookup order?** GUI applications (e.g. En Croissant on macOS) launch the engine process from an arbitrary working directory — often the app bundle root. Resolving the path relative to `current_exe()` is the only portable way to guarantee the file is found regardless of CWD.

---

## Supported UCI commands

| Command | Engine response | Notes |
|---|---|---|
| `uci` | `id name HybridNet` · `id author serkan` · `uciok` | Handshake |
| `isready` | `readyok` | Synchronisation ping |
| `ucinewgame` | *(none)* | Resets board to starting position |
| `position startpos [moves …]` | *(none)* | Sets board; replays move list |
| `position fen <FEN> [moves …]` | *(none)* | Sets board from FEN; replays move list |
| `go [any options]` | `info …` · `bestmove <move>` | Selects and returns best move |
| `quit` | *(exits)* | Terminates the process |

All options passed to `go` (depth, time controls, etc.) are silently ignored — the engine always does a single 1-ply search.

### `go` output

```
info depth 1 score cp <N> pv <move>
bestmove <move>
```

`score cp` is the position evaluation in centipawns, converted from the model's raw output in `(−1, +1)` using `eval × 1000`. Positive values favour White.

If no legal moves exist (checkmate or stalemate) the engine responds with `bestmove 0000`.

---

## Move selection algorithm (`search.rs`)

The engine uses **greedy 1-ply search** — no tree, no Monte Carlo, no iterative deepening.

```
for every legal move in the current position:
    apply the move → get the resulting board
evaluate all resulting boards in one batched forward pass through HybridNet
pick the move whose resulting position has the highest value
    from the current player's perspective
```

### Sign convention

`HybridNet` always outputs a value in `(−1, +1)` from **White's perspective**:
- `+1.0` → White is winning
- `−1.0` → Black is winning

Move selection multiplies each raw value by a sign before comparing:

```rust
let sign = match board.side_to_move {
    Color::White =>  1.0_f32,   // maximise raw value
    Color::Black => -1.0_f32,   // minimise raw value = maximise negated value
};
```

The selected move and its signed score are returned together:

```rust
pub fn best_move(model: &HybridValueNet, board: &Board) -> Option<(Move, f32)>
```

The returned score is already from the **current player's perspective** (always positive when winning), which is what the GUI-facing centipawn conversion expects.

### Batch inference

All successor positions are evaluated in a **single call** to `model.forward_batch()`, which is substantially faster than calling `model.forward()` once per legal move. At the starting position this means one batch of 20 positions; mid-game batches are typically 20–40.

---

## Move parsing (`parse_uci_move`)

Converts a UCI long-algebraic string (e.g. `"e2e4"`, `"e7e8q"`) into a `Move` by matching against the legal move list of the given position.

```rust
pub fn parse_uci_move(board: &Board, s: &str) -> Option<Move>
```

Matching against legal moves is required because the `Move` type carries a `MoveKind` field (`Normal`, `Castling`, `EnPassant`) that cannot be inferred from the from/to squares alone. Constructing a `Move` directly would silently produce an incorrect kind for castling and en passant.

**Rejection rules:**
- Fewer than 4 ASCII characters → `None`
- Non-ASCII bytes → `None` (avoids panicking on multi-byte UTF-8 slice indexing)
- 5th character present but not one of `q r b n` → `None` (unknown promotion piece)
- Move not found in the legal move list → `None` (illegal move)

---

## Tracing / logging

All log output goes to **stderr** so it never pollutes the UCI stdout channel. Log level is controlled by the `RUST_LOG` environment variable (default: `info`).

```bash
RUST_LOG=debug target/release/uci
```

Useful log events:
- `WARN  no saved model` — model.bin not found; engine uses random weights
- `WARN  invalid FEN` — GUI sent a malformed FEN; engine falls back to starting position
- `DEBUG game complete` — emitted per ply during self-play (not during UCI play)

---

## Tests

```bash
cargo test -p uci
```

| Test | What it checks |
|---|---|
| `parse_e2e4_from_start` | Normal pawn move parsed correctly |
| `parse_promotion_move` | Queen promotion `e7e8q` returns correct piece kind |
| `parse_illegal_move_returns_none` | Illegal move (e2e5) returns `None` |
| `parse_malformed_returns_none` | Short/empty strings return `None` |
| `parse_unknown_promo_char_returns_none` | `e7e8x` returns `None` |
| `best_move_returns_legal_move` | Selected move is in the legal move list; score is finite |
| `best_move_returns_none_when_no_legal_moves` | Fool's Mate position returns `None` |
