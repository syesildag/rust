# Chess Bitboard — Implementation Plan

## Context

The workspace (`/Users/serkan/Workspace/rust`) is a Cargo workspace with two crates: `core` (general-purpose library) and `cli` (binary). The goal is to add a `crates/chess` library crate implementing a pure bitboard chess engine, then wire `cli` up as the chess application. The chess library must support FEN parsing, fully legal move generation (including castling, en passant, promotion), move validation, and game state detection (checkmate, stalemate, draw conditions).

---

## Architecture

```
crates/
├── core/       — shared utilities (existing, untouched for now)
├── chess/      — new chess library (bitboard engine)
│   └── Cargo.toml  depends on: nothing external
└── cli/        — chess app binary
    └── Cargo.toml  depends on: chess = { path = "../chess" }
```

Root `Cargo.toml` workspace `members` gains `"crates/chess"`.

---

## Files to Create

### `crates/chess/Cargo.toml`
Inherits workspace version/edition/lints. No external dependencies.

### `crates/chess/src/lib.rs`
Public re-exports. Doc example: parse starting FEN, generate 20 moves.

### `crates/chess/src/square.rs`
- `Square(u8)` newtype, 0–63 (a1=0, h8=63)
- `Square::from_index(u8)`, `Square::from_algebraic(&str)`
- `Square::file(self) -> u8`, `Square::rank(self) -> u8`
- `impl Display` → `"e4"` format

### `crates/chess/src/piece.rs`
- `enum Color { White, Black }` with `fn opposite(self)`
- `enum PieceKind { Pawn, Knight, Bishop, Rook, Queen, King }`
- `struct Piece { kind: PieceKind, color: Color }`

### `crates/chess/src/bitboard.rs`
- `type Bitboard = u64`
- Helper fns: `set_bit`, `clear_bit`, `lsb_square`, `pop_lsb`, `count_bits`
- `impl Display` for debug 8×8 grid printing

### `crates/chess/src/attack.rs`
Precomputed lookup tables (computed in `fn init()` called once at startup via `std::sync::OnceLock`):
- `KNIGHT_ATTACKS: [u64; 64]`
- `KING_ATTACKS: [u64; 64]`
- `PAWN_ATTACKS: [[u64; 64]; 2]`  — index by `[color][square]`
- `RAY_ATTACKS: [[u64; 8]; 64]`   — 8 directions (N/NE/E/SE/S/SW/W/NW)

Sliding piece attack fn:
```rust
fn sliding_attacks(sq: Square, occupied: u64, dirs: &[usize]) -> u64
```
Uses classical o^(o-2r) subtraction trick per positive ray; bit-reverse for negative rays.

### `crates/chess/src/board.rs`
```rust
pub struct Board {
    pub pieces: [[u64; 6]; 2],      // [Color as usize][PieceKind as usize]
    pub side_to_move: Color,
    pub castling: u8,               // bits: 0=WK, 1=WQ, 2=BK, 3=BQ
    pub en_passant: Option<Square>,
    pub halfmove_clock: u8,
    pub fullmove_number: u16,
}
```

Methods:
- `Board::starting_position() -> Board`  (delegates to FEN)
- `Board::white_occupied(&self) -> u64`
- `Board::black_occupied(&self) -> u64`
- `Board::all_occupied(&self) -> u64`
- `Board::piece_at(&self, sq: Square) -> Option<Piece>`
- `Board::make_move(&self, mv: Move) -> Board`  — immutable, returns new Board
- `Board::is_in_check(&self, color: Color) -> bool`
- `Board::to_fen(&self) -> String`

`make_move` steps:
1. Clear `from`, set `to` in piece bitboard
2. Clear enemy bit at `to` if capture
3. En passant: also clear pawn on rank of `from`, file of `to`
4. Castling: move rook (a1/h1/a8/h8 → d1/f1/d8/f8)
5. Promotion: clear pawn bit, set chosen piece bit at `to`
6. Update castling rights (king/rook moved from starting square)
7. Update en passant (set only on double pawn push, else `None`)
8. Halfmove clock: reset on pawn move or capture; else +1
9. Fullmove: increment after Black's move

### `crates/chess/src/moves.rs`
```rust
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceKind>,
    pub kind: MoveKind,
}

pub enum MoveKind { Normal, Castling, EnPassant }
```

### `crates/chess/src/movegen.rs`
```rust
pub fn generate_legal_moves(board: &Board) -> Vec<Move>
fn generate_pseudo_legal(board: &Board) -> Vec<Move>
fn pawn_moves(board: &Board) -> Vec<Move>
fn knight_moves(board: &Board) -> Vec<Move>
fn bishop_moves(board: &Board) -> Vec<Move>
fn rook_moves(board: &Board) -> Vec<Move>
fn queen_moves(board: &Board) -> Vec<Move>
fn king_moves(board: &Board) -> Vec<Move>   // includes castling
```

`generate_legal_moves`: calls `generate_pseudo_legal`, then filters with:
```rust
board.make_move(mv).is_in_check(board.side_to_move) == false
```

Castling additional checks: king not currently in check, king's transit square not attacked, destination not attacked.

### `crates/chess/src/fen.rs`
```rust
pub enum FenError {
    WrongFieldCount,
    InvalidPieceChar(char),
    InvalidSquare(String),
    InvalidCastling(String),
    InvalidSideToMove(char),
    ParseInt(std::num::ParseIntError),
}

pub fn from_fen(s: &str) -> Result<Board, FenError>
```

Parsing order: piece placement (ranks 8→1) → side → castling → en passant → halfmove → fullmove.

### `crates/chess/src/game.rs`
```rust
pub enum DrawReason { FiftyMoveRule, InsufficientMaterial }
pub enum GameStatus { Ongoing, Checkmate, Stalemate, Draw(DrawReason) }

pub fn game_status(board: &Board) -> GameStatus
```

Logic:
1. Check 50-move rule first (`halfmove_clock >= 100`)
2. Check insufficient material
3. Generate legal moves — if empty: checkmate (in check) or stalemate (not in check)
4. Otherwise: `Ongoing`

Insufficient material patterns:
- Both sides: king only
- One side king only, other has only knight or only bishop

### `crates/cli/src/main.rs` (replace)
Accept FEN from args, print all legal moves + game status.

---

## Files to Modify

| File | Change |
|---|---|
| `Cargo.toml` (root) | Add `"crates/chess"` to `members` |
| `crates/cli/Cargo.toml` | Add `chess = { path = "../chess" }` dependency |
| `crates/cli/src/main.rs` | Replace add-two-numbers logic with chess CLI |

---

## Test Plan

### Unit tests (inline `#[cfg(test)]`)

| File | Tests |
|---|---|
| `square.rs` | `from_algebraic("e4")` → index 28; `file`/`rank` accessors |
| `bitboard.rs` | `lsb_square`, `pop_lsb`, `count_bits` on known values |
| `attack.rs` | Knight on a1 has 2 attacks; knight on d4 has 8; king on e1 correct mask |
| `fen.rs` | Starting FEN round-trips; error on invalid piece char |
| `board.rs` | `piece_at` after `make_move`; en passant square updated correctly |

### Move generation tests

| Test | Position (FEN) | Expected |
|---|---|---|
| Starting position | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` | 20 legal moves |
| Perft depth 2 | same | 400 nodes |
| Perft depth 3 | same | 8902 nodes |
| En passant | `rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3` | en passant to f6 included |
| Castling | `r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1` | both kingside and queenside castling |
| Promotion | `8/P7/8/8/8/8/8/4K2k w - - 0 1` | 4 promotion moves |
| Pin detection | `4k3/8/8/8/r7/8/4R3/4K3 w - - 0 1` | rook on e2 is pinned, cannot move off e-file |

### Game state tests

| Test | FEN | Expected |
|---|---|---|
| Scholar's mate | `rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3` | `Checkmate` |
| Stalemate | `k7/8/1Q6/8/8/8/8/7K b - - 0 1` | `Stalemate` |
| 50-move draw | Starting FEN with halfmove_clock=100 | `Draw(FiftyMoveRule)` |
| Insufficient material | `4k3/8/8/8/8/8/8/4K3 w - - 0 1` | `Draw(InsufficientMaterial)` |

---

## Build Order

1. `square.rs` + `piece.rs` (no deps)
2. `bitboard.rs` (no deps)
3. `attack.rs` (depends on square, bitboard)
4. `moves.rs` (depends on square, piece)
5. `fen.rs` + `board.rs` (depends on all above)
6. `movegen.rs` (depends on board, attack, moves)
7. `game.rs` (depends on movegen, board)
8. `lib.rs` re-exports
9. `cli/main.rs` wiring
10. Tests at each layer

---

## Verification

```bash
cargo test -p chess              # all unit + integration tests
cargo test -p chess perft        # perft tests specifically
cargo run -p cli -- "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# expected: 20 legal moves listed, status: Ongoing
cargo check-all                  # fmt + clippy must pass clean
```
