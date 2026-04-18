use std::fmt;

use crate::piece::PieceKind;
use crate::square::Square;

/// Whether a move is a normal move, castling, or en passant capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveKind {
    Normal,
    Castling,
    EnPassant,
}

/// A chess move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceKind>,
    pub kind: MoveKind,
}

impl Move {
    #[must_use]
    pub const fn normal(from: Square, to: Square) -> Self {
        Self {
            from,
            to,
            promotion: None,
            kind: MoveKind::Normal,
        }
    }

    #[must_use]
    pub const fn promotion(from: Square, to: Square, piece: PieceKind) -> Self {
        Self {
            from,
            to,
            promotion: Some(piece),
            kind: MoveKind::Normal,
        }
    }

    #[must_use]
    pub const fn castling(from: Square, to: Square) -> Self {
        Self {
            from,
            to,
            promotion: None,
            kind: MoveKind::Castling,
        }
    }

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
