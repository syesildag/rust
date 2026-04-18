# Crates

This workspace has three crates. Each crate has its own `Cargo.toml` manifest but inherits
shared configuration (version, edition, lint rules) from the workspace root.

---

## `core` — Shared Utilities Library

**Location:** [`crates/core/`](../crates/core/)
**Entry point:** [`crates/core/src/lib.rs`](../crates/core/src/lib.rs)

A general-purpose library crate. It has no `main` function — it exposes public functions
that other crates in the workspace can import and use. It is intentionally domain-agnostic
so it can serve as a foundation for any future crate.

### Current API

#### `add(a: i32, b: i32) -> i32`

Adds two integers and returns the result.

```rust
use core::add;

let sum = add(2, 3); // → 5
```

---

## `chess` — Chess Engine Library

**Location:** [`crates/chess/`](../crates/chess/)
**Entry point:** [`crates/chess/src/lib.rs`](../crates/chess/src/lib.rs)

A pure bitboard chess engine library. It has no `main` — all logic is exposed through a
public API consumed by `cli`. See [chess-engine.md](chess-engine.md) for a full technical
reference.

### Modules

| Module | Responsibility |
|---|---|
| `square` | `Square(u8)` newtype, algebraic notation parsing |
| `piece` | `Color`, `PieceKind`, `Piece` enums and types |
| `bitboard` | `u64` alias + helpers: `lsb_square`, `pop_lsb`, `count_bits` |
| `attack` | Precomputed attack tables, sliding piece ray attacks |
| `moves` | `Move` struct with `MoveKind` (Normal / Castling / EnPassant) |
| `board` | `Board` state, `make_move`, `is_in_check` |
| `fen` | `from_fen` / `to_fen`, `FenError` |
| `movegen` | Full legal move generation, `perft` |
| `game` | `GameStatus` detection: checkmate, stalemate, draw |

### Quick start

```rust
use chess::board::Board;
use chess::movegen::generate_legal_moves;
use chess::game::game_status;

let board = Board::starting_position();
let moves = generate_legal_moves(&board);
assert_eq!(moves.len(), 20);
println!("Status: {:?}", game_status(&board));
```

---

## `cli` — Chess Application Binary

**Location:** [`crates/cli/`](../crates/cli/)
**Entry point:** [`crates/cli/src/main.rs`](../crates/cli/src/main.rs)

The runnable chess application. Its only job is to wire together `chess` (and optionally
`core`) and present results to the user. All game logic lives in `chess`.

### Usage

```bash
# Starting position
cargo run -p cli

# Custom position via FEN (6 fields as separate arguments)
cargo run -p cli -- rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

### Example output

```
FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Side to move: White
Legal moves (20): a2a3, a2a4, b2b3, b2b4, ...
Status: Ongoing
```

### Dependencies

```toml
[dependencies]
core  = { path = "../core" }
chess = { path = "../chess" }
```

---

## Adding a New Crate

1. Create `crates/<name>/src/` and a `Cargo.toml` that inherits workspace settings
2. Add `"crates/<name>"` to the `members` list in the root `Cargo.toml`
3. Reference `chess` or `core` via `{ path = "../chess" }` as needed
