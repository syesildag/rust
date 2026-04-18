# Chess Engine — Technical Reference

The `chess` crate is a pure bitboard chess engine written in safe Rust. It supports full
legal move generation (including castling, en passant, and promotion), FEN parsing, and
game state detection (checkmate, stalemate, draw conditions).

---

## Board Representation

### Bitboards

The board is represented as 12 `u64` values — one per `(Color, PieceKind)` pair:

```
pieces: [[u64; 6]; 2]   // [color][piece_kind]
```

Each `u64` is a **bitboard**: bit `i` is set when a piece of that type occupies square `i`.

```
Square index layout (a1 = 0, h8 = 63):

  a  b  c  d  e  f  g  h
8 56 57 58 59 60 61 62 63
7 48 49 50 51 52 53 54 55
6 40 41 42 43 44 45 46 47
5 32 33 34 35 36 37 38 39
4 24 25 26 27 28 29 30 31
3 16 17 18 19 20 21 22 23
2  8  9 10 11 12 13 14 15
1  0  1  2  3  4  5  6  7
```

Three derived boards are computed on the fly: `white_occupied`, `black_occupied`,
`all_occupied` (bitwise OR of the relevant 6 bitboards).

### Full game state (`Board`)

```rust
pub struct Board {
    pub pieces: [[u64; 6]; 2],   // [Color][PieceKind]
    pub side_to_move: Color,
    pub castling: u8,            // bits: 0=WK, 1=WQ, 2=BK, 3=BQ
    pub en_passant: Option<Square>,
    pub halfmove_clock: u8,
    pub fullmove_number: u16,
}
```

`Board` is an immutable value type — `make_move` returns a new `Board` rather than
modifying in place. This makes it trivial to explore move trees without undo logic.

---

## Attack Tables

All attack tables are precomputed once at startup using `std::sync::OnceLock`:

| Table | Size | Contents |
|---|---|---|
| `KNIGHT_ATTACKS` | `[u64; 64]` | Attack mask for a knight on each square |
| `KING_ATTACKS` | `[u64; 64]` | Attack mask for a king on each square |
| `PAWN_ATTACKS` | `[[u64; 64]; 2]` | Attack mask per color per square |
| `RAY_ATTACKS` | `[[u64; 8]; 64]` | One full ray per direction per square |

### Ray directions

Directions are numbered so that the first 4 (0–3) are **positive** (bit index increases
along the ray) and the last 4 (4–7) are **negative** (bit index decreases):

```
N=0  NE=1  E=2  NW=3   (positive)
S=4  SW=5  W=6  SE=7   (negative)
```

### Sliding piece attacks — o^(o-2r) trick

For positive rays, sliding piece attacks are computed using the classical formula:

```
o = occupied & ray
attacks = (o - 2r) ^ o,  masked with ray
```

Where `r` is the piece's square bit. This propagates a borrow through the ray, stopping
at the first blocker (inclusive).

For negative rays, all bits are reversed, the same formula is applied, then reversed back.
This avoids needing magic bitboards while remaining efficient.

---

## Move Generation

### Pipeline

```
generate_legal_moves(board)
  1. generate_pseudo_legal(board)   — fast bitboard ops, ignores check
  2. for each move: board.make_move(mv).is_in_check(side_to_move)
  3. filter out any move that leaves own king in check
```

### Piece-specific generation

| Piece | Strategy |
|---|---|
| Pawn | Single/double push, diagonal captures, en passant, promotion (×4 on 7th rank) |
| Knight | `KNIGHT_ATTACKS[sq] & !own_occupied` |
| Bishop | Ray attacks on NE/NW/SW/SE diagonals |
| Rook | Ray attacks on N/E/S/W axes |
| Queen | Bishop + Rook combined |
| King | `KING_ATTACKS[sq] & !own_occupied` + castling |

### Castling validation

Beyond having the castling rights set, castling is only generated when:
1. The king is **not currently in check**
2. All squares between king and rook are **unoccupied**
3. The squares the king **passes through** are not attacked by the enemy

### Pin detection

Pins are handled implicitly: every pseudo-legal move is tested by applying it and calling
`is_in_check`. A pinned piece's illegal moves are filtered out because they leave the king
in check. This is correct but not the fastest possible approach — a future optimisation
would pre-compute a pin bitmask to avoid the board clone for most moves.

---

## Check Detection (`is_in_check`)

`is_in_check(color)` locates the king square, then tests whether any enemy piece attacks it:

1. **Knight** — `KNIGHT_ATTACKS[king_sq] & enemy_knights`
2. **Pawn** — `PAWN_ATTACKS[color][king_sq] & enemy_pawns`
3. **King** — `KING_ATTACKS[king_sq] & enemy_king` (prevents kings touching)
4. **Rook/Queen** — `rook_attacks(king_sq, occupied) & (enemy_rooks | enemy_queens)`
5. **Bishop/Queen** — `bishop_attacks(king_sq, occupied) & (enemy_bishops | enemy_queens)`

---

## FEN Support

FEN (Forsyth-Edwards Notation) is the standard format for encoding a chess position.

### Parsing

```rust
let board = chess::fen::from_fen(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
)?;
```

Fields parsed in order: piece placement → side to move → castling rights →
en passant square → halfmove clock → fullmove number.

### Serialisation

```rust
let fen_string = board.to_fen();
```

The round-trip `from_fen(to_fen(board))` always produces an identical board.

### Errors

`FenError` is a typed enum, not a string:

| Variant | Cause |
|---|---|
| `WrongFieldCount(n)` | FEN does not have exactly 6 space-separated fields |
| `InvalidPieceChar(c)` | Unrecognised character in piece placement |
| `InvalidSquare(s)` | Malformed square in en passant field |
| `InvalidCastling(s)` | Unrecognised castling rights string |
| `InvalidSideToMove(c)` | Side-to-move field is not `w` or `b` |
| `ParseInt(e)` | Halfmove clock or fullmove number is not a valid integer |

---

## Board Display

`Board` implements `fmt::Display`, so it can be printed directly with `println!` or
`format!` — no helper function needed.

```rust
println!("{}", Board::starting_position());
// or equivalently:
println!("{board}");
```

Output format (rank 8 at top, rank 1 at bottom):

```
  a b c d e f g h
8 r n b q k b n r 8
7 p p p p p p p p 7
6 · · · · · · · · 6
5 · · · · · · · · 5
4 · · · · · · · · 4
3 · · · · · · · · 3
2 P P P P P P P P 2
1 R N B Q K B N R 1
  a b c d e f g h
```

Conventions:
- **Uppercase** letters — white pieces (R N B Q K P)
- **Lowercase** letters — black pieces (r n b q k p)
- `·` — empty square

---

## Game Status

```rust
pub enum GameStatus {
    Ongoing,
    Checkmate,
    Stalemate,
    Draw(DrawReason),
}

pub enum DrawReason {
    FiftyMoveRule,
    InsufficientMaterial,
}
```

`game_status(board)` checks in this order:

1. **50-move rule** — `halfmove_clock >= 100`
2. **Insufficient material** — kings only; king + bishop vs king; king + knight vs king
3. **No legal moves + in check** → Checkmate
4. **No legal moves + not in check** → Stalemate
5. Otherwise → Ongoing

---

## Perft — Move Generation Correctness

`perft(board, depth)` counts all leaf nodes at exactly `depth` plies. Known correct values
from the starting position are used as regression tests:

| Depth | Nodes |
|---|---|
| 1 | 20 |
| 2 | 400 |
| 3 | 8,902 |

If `perft(3)` returns exactly 8902, the full move generator (including all special moves)
is correct for the starting position.

---

## Public API Summary

```rust
// Board construction
Board::starting_position() -> Board
chess::fen::from_fen(s: &str) -> Result<Board, FenError>

// Board queries
board.piece_at(sq: Square) -> Option<Piece>
board.is_in_check(color: Color) -> bool
board.to_fen() -> String
fmt::Display for Board  // println!("{board}") prints ASCII grid

// Move generation
chess::movegen::generate_legal_moves(board: &Board) -> Vec<Move>
chess::movegen::perft(board: &Board, depth: u32) -> u64

// Game state
chess::game::game_status(board: &Board) -> GameStatus

// Move application (immutable — returns new Board)
board.make_move(mv: Move) -> Board
```
