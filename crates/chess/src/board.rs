use crate::attack;
use crate::bitboard::{clear_bit, lsb_square, set_bit};
use crate::fen;
use crate::moves::{Move, MoveKind};
use crate::piece::{Color, Piece, PieceKind};
use crate::square::Square;

/// Castling right bit masks (field `Board::castling`).
pub const WK: u8 = 1; // White kingside
pub const WQ: u8 = 2; // White queenside
pub const BK: u8 = 4; // Black kingside
pub const BQ: u8 = 8; // Black queenside

/// The complete game state represented as 12 bitboards plus auxiliary fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    /// `pieces[color][piece_kind]` — bitboard of that piece type for that color.
    pub pieces: [[u64; 6]; 2],
    pub side_to_move: Color,
    /// Castling availability: bits 0=WK, 1=WQ, 2=BK, 3=BQ.
    pub castling: u8,
    /// En passant target square (the square the capturing pawn moves *to*).
    pub en_passant: Option<Square>,
    pub halfmove_clock: u8,
    pub fullmove_number: u16,
}

impl Board {
    /// Returns the board for the standard chess starting position.
    ///
    /// # Panics
    /// Panics if the hard-coded FEN is malformed (should never happen).
    #[must_use]
    pub fn starting_position() -> Self {
        fen::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("starting FEN is always valid")
    }

    /// Returns the combined bitboard of all white pieces.
    #[must_use]
    pub fn white_occupied(&self) -> u64 {
        self.pieces[Color::White as usize]
            .iter()
            .fold(0, |a, &b| a | b)
    }

    /// Returns the combined bitboard of all black pieces.
    #[must_use]
    pub fn black_occupied(&self) -> u64 {
        self.pieces[Color::Black as usize]
            .iter()
            .fold(0, |a, &b| a | b)
    }

    /// Returns the combined bitboard of all pieces.
    #[must_use]
    pub fn all_occupied(&self) -> u64 {
        self.white_occupied() | self.black_occupied()
    }

    /// Returns the piece on `sq`, if any.
    #[must_use]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        let bit = sq.bit();
        for (ci, color) in [Color::White, Color::Black].iter().enumerate() {
            for kind in PieceKind::ALL {
                if self.pieces[ci][kind.index()] & bit != 0 {
                    return Some(Piece::new(kind, *color));
                }
            }
        }
        None
    }

    /// Returns `true` if `color`'s king is currently in check.
    #[must_use]
    pub fn is_in_check(&self, color: Color) -> bool {
        let king_bb = self.pieces[color as usize][PieceKind::King.index()];
        if king_bb == 0 {
            return false;
        }
        let king_sq = lsb_square(king_bb);
        let occupied = self.all_occupied();
        let enemy = color.opposite();
        let ei = enemy as usize;

        // Knight attacks
        if attack::knight_attacks(king_sq) & self.pieces[ei][PieceKind::Knight.index()] != 0 {
            return true;
        }
        // Pawn attacks
        if attack::pawn_attacks(color, king_sq) & self.pieces[ei][PieceKind::Pawn.index()] != 0 {
            return true;
        }
        // King attacks (used to prevent kings from touching)
        if attack::king_attacks(king_sq) & self.pieces[ei][PieceKind::King.index()] != 0 {
            return true;
        }
        // Rook/queen on same rank/file
        let rq =
            self.pieces[ei][PieceKind::Rook.index()] | self.pieces[ei][PieceKind::Queen.index()];
        if attack::rook_attacks(king_sq, occupied) & rq != 0 {
            return true;
        }
        // Bishop/queen on same diagonal
        let bq =
            self.pieces[ei][PieceKind::Bishop.index()] | self.pieces[ei][PieceKind::Queen.index()];
        if attack::bishop_attacks(king_sq, occupied) & bq != 0 {
            return true;
        }

        false
    }

    /// Applies `mv` and returns the resulting board. Does not validate legality.
    ///
    /// # Panics
    /// Panics if `mv.from` does not contain a piece belonging to the side to move.
    #[must_use]
    pub fn make_move(&self, mv: Move) -> Self {
        let mut next = self.clone();
        let color = self.side_to_move;
        let ci = color as usize;
        let ei = color.opposite() as usize;
        let occupied = self.all_occupied();

        // Find the moving piece kind
        let moving_kind = self
            .piece_kind_at(mv.from)
            .expect("no piece on from square");

        // Clear from square
        next.pieces[ci][moving_kind.index()] =
            clear_bit(next.pieces[ci][moving_kind.index()], mv.from);

        // Handle captures: clear any enemy piece on the to square
        for kind in PieceKind::ALL {
            next.pieces[ei][kind.index()] = clear_bit(next.pieces[ei][kind.index()], mv.to);
        }

        match mv.kind {
            MoveKind::Normal => {
                let land_kind = mv.promotion.unwrap_or(moving_kind);
                next.pieces[ci][land_kind.index()] =
                    set_bit(next.pieces[ci][land_kind.index()], mv.to);
            }
            MoveKind::EnPassant => {
                // The captured pawn is on the same rank as `from`, same file as `to`
                let captured_sq = Square::from_file_rank(mv.to.file(), mv.from.rank());
                next.pieces[ei][PieceKind::Pawn.index()] =
                    clear_bit(next.pieces[ei][PieceKind::Pawn.index()], captured_sq);
                next.pieces[ci][PieceKind::Pawn.index()] =
                    set_bit(next.pieces[ci][PieceKind::Pawn.index()], mv.to);
            }
            MoveKind::Castling => {
                next.pieces[ci][PieceKind::King.index()] =
                    set_bit(next.pieces[ci][PieceKind::King.index()], mv.to);
                // Move the rook
                let (rook_from, rook_to) = castling_rook_squares(color, mv.to);
                next.pieces[ci][PieceKind::Rook.index()] =
                    clear_bit(next.pieces[ci][PieceKind::Rook.index()], rook_from);
                next.pieces[ci][PieceKind::Rook.index()] =
                    set_bit(next.pieces[ci][PieceKind::Rook.index()], rook_to);
            }
        }

        // Update castling rights when king or rook moves
        next.castling &= castling_rights_mask(mv.from) & castling_rights_mask(mv.to);

        // Update en passant square
        next.en_passant =
            if moving_kind == PieceKind::Pawn && mv.to.rank().abs_diff(mv.from.rank()) == 2 {
                // Double pawn push: target square is between from and to
                let ep_rank = u8::midpoint(mv.from.rank(), mv.to.rank());
                Some(Square::from_file_rank(mv.from.file(), ep_rank))
            } else {
                None
            };

        // Halfmove clock
        let is_capture = occupied & mv.to.bit() != 0 || mv.kind == MoveKind::EnPassant;
        if moving_kind == PieceKind::Pawn || is_capture {
            next.halfmove_clock = 0;
        } else {
            next.halfmove_clock = next.halfmove_clock.saturating_add(1);
        }

        // Fullmove number increments after Black's move
        if color == Color::Black {
            next.fullmove_number = next.fullmove_number.saturating_add(1);
        }

        next.side_to_move = color.opposite();
        next
    }

    /// Returns the FEN string for this board.
    #[must_use]
    pub fn to_fen(&self) -> String {
        fen::to_fen(self)
    }

    /// Returns the piece kind on `sq` for the moving side, if any.
    fn piece_kind_at(&self, sq: Square) -> Option<PieceKind> {
        let bit = sq.bit();
        PieceKind::ALL
            .into_iter()
            .find(|&kind| self.pieces[self.side_to_move as usize][kind.index()] & bit != 0)
    }
}

/// Returns the squares a rook moves between during castling.
/// `king_to` is the king's destination square (g1/c1/g8/c8).
fn castling_rook_squares(color: Color, king_to: Square) -> (Square, Square) {
    match (color, king_to.file()) {
        (Color::White, 6) => (
            Square::from_algebraic("h1").unwrap(),
            Square::from_algebraic("f1").unwrap(),
        ),
        (Color::White, _) => (
            Square::from_algebraic("a1").unwrap(),
            Square::from_algebraic("d1").unwrap(),
        ),
        (Color::Black, 6) => (
            Square::from_algebraic("h8").unwrap(),
            Square::from_algebraic("f8").unwrap(),
        ),
        (Color::Black, _) => (
            Square::from_algebraic("a8").unwrap(),
            Square::from_algebraic("d8").unwrap(),
        ),
    }
}

/// Bitmask of castling rights to remove when a piece moves from/to this square.
const fn castling_rights_mask(sq: Square) -> u8 {
    match sq.index() {
        4 => !(WK | WQ),  // e1 — white king moved
        0 => !WQ,         // a1 — white queenside rook
        7 => !WK,         // h1 — white kingside rook
        60 => !(BK | BQ), // e8 — black king moved
        56 => !BQ,        // a8 — black queenside rook
        63 => !BK,        // h8 — black kingside rook
        _ => 0xFF,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starting_position_piece_count() {
        let board = Board::starting_position();
        assert_eq!(board.all_occupied().count_ones(), 32);
        assert_eq!(board.white_occupied().count_ones(), 16);
        assert_eq!(board.black_occupied().count_ones(), 16);
    }

    #[test]
    fn piece_at_starting_position() {
        let board = Board::starting_position();
        let e1 = Square::from_algebraic("e1").unwrap();
        let e8 = Square::from_algebraic("e8").unwrap();
        assert_eq!(
            board.piece_at(e1),
            Some(Piece::new(PieceKind::King, Color::White))
        );
        assert_eq!(
            board.piece_at(e8),
            Some(Piece::new(PieceKind::King, Color::Black))
        );
    }

    #[test]
    fn not_in_check_at_start() {
        let board = Board::starting_position();
        assert!(!board.is_in_check(Color::White));
        assert!(!board.is_in_check(Color::Black));
    }

    #[test]
    fn make_move_e2_e4() {
        let board = Board::starting_position();
        let mv = Move::normal(
            Square::from_algebraic("e2").unwrap(),
            Square::from_algebraic("e4").unwrap(),
        );
        let next = board.make_move(mv);
        assert!(next
            .piece_at(Square::from_algebraic("e2").unwrap())
            .is_none());
        assert_eq!(
            next.piece_at(Square::from_algebraic("e4").unwrap()),
            Some(Piece::new(PieceKind::Pawn, Color::White))
        );
        assert_eq!(next.side_to_move, Color::Black);
        assert_eq!(next.en_passant, Square::from_algebraic("e3"));
    }

    #[test]
    fn en_passant_cleared_after_non_pawn_move() {
        let board = Board::starting_position();
        let mv = Move::normal(
            Square::from_algebraic("e2").unwrap(),
            Square::from_algebraic("e4").unwrap(),
        );
        let board = board.make_move(mv);
        // Black plays a7-a6
        let mv2 = Move::normal(
            Square::from_algebraic("a7").unwrap(),
            Square::from_algebraic("a6").unwrap(),
        );
        let board = board.make_move(mv2);
        assert!(board.en_passant.is_none());
    }
}
