use std::fmt;

use crate::piece::PieceKind;
use crate::square::Square;

/// Whether a move is a normal move, castling, or en passant capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveKind {
    /// A standard move or capture, including pawn pushes and promotions.
    Normal,
    /// A king-side or queen-side castling move (king moves two squares, rook relocates).
    Castling,
    /// An en passant pawn capture (the captured pawn is not on the destination square).
    EnPassant,
}

/// A chess move with origin, destination, optional promotion, and move kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    /// The square the moving piece departs from.
    pub from: Square,
    /// The square the moving piece (or the promoted piece) lands on.
    pub to: Square,
    /// Non-`None` for pawn promotions; the piece kind the pawn promotes to.
    pub promotion: Option<PieceKind>,
    /// Classifies the move for special handling in `make_move`.
    pub kind: MoveKind,
}

impl Move {
    /// Creates a normal move or capture (no promotion, not castling or en passant).
    #[must_use]
    pub const fn normal(from: Square, to: Square) -> Self {
        Self {
            from,
            to,
            promotion: None,
            kind: MoveKind::Normal,
        }
    }

    /// Creates a promotion move; `piece` is the piece kind the pawn promotes to.
    #[must_use]
    pub const fn promotion(from: Square, to: Square, piece: PieceKind) -> Self {
        Self {
            from,
            to,
            promotion: Some(piece),
            kind: MoveKind::Normal,
        }
    }

    /// Creates a castling move; `from` is the king's origin and `to` is its destination (g1/c1/g8/c8).
    #[must_use]
    pub const fn castling(from: Square, to: Square) -> Self {
        Self {
            from,
            to,
            promotion: None,
            kind: MoveKind::Castling,
        }
    }

    /// Creates an en passant capture; `to` is the target square (where the pawn lands, not where the captured pawn was).
    #[must_use]
    pub const fn en_passant(from: Square, to: Square) -> Self {
        Self {
            from,
            to,
            promotion: None,
            kind: MoveKind::EnPassant,
        }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from, self.to)?;
        if let Some(promo) = self.promotion {
            write!(f, "{}", promo.fen_char().to_ascii_lowercase())?;
        }
        Ok(())
    }
}
