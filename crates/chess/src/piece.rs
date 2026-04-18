/// Side to move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    White,
    Black,
}

impl Color {
    /// Returns the opposite color.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }
}

/// The type of a chess piece, without color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceKind {
    /// All piece kinds in a stable order (matches bitboard array indexing).
    pub const ALL: [Self; 6] = [
        Self::Pawn,
        Self::Knight,
        Self::Bishop,
        Self::Rook,
        Self::Queen,
        Self::King,
    ];

    /// Index into a `[T; 6]` array for this piece kind.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Pawn => 0,
            Self::Knight => 1,
            Self::Bishop => 2,
            Self::Rook => 3,
            Self::Queen => 4,
            Self::King => 5,
        }
    }

    /// FEN character for this piece (uppercase = white convention).
    #[must_use]
    pub const fn fen_char(self) -> char {
        match self {
            Self::Pawn => 'P',
            Self::Knight => 'N',
            Self::Bishop => 'B',
            Self::Rook => 'R',
            Self::Queen => 'Q',
            Self::King => 'K',
        }
    }
}

/// A chess piece with a kind and a color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
}

impl Piece {
    #[must_use]
    pub const fn new(kind: PieceKind, color: Color) -> Self {
        Self { kind, color }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opposite_color() {
        assert_eq!(Color::White.opposite(), Color::Black);
        assert_eq!(Color::Black.opposite(), Color::White);
    }

    #[test]
    fn piece_kind_indices_unique() {
        let mut seen = [false; 6];
        for kind in PieceKind::ALL {
            let i = kind.index();
            assert!(!seen[i], "duplicate index {i}");
            seen[i] = true;
        }
    }
}
