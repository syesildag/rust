use std::fmt;

/// A chess square, represented as an index 0–63 (a1=0, b1=1, …, h8=63).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Square(u8);

impl Square {
    /// Creates a square from a 0–63 index without bounds checking.
    #[must_use]
    pub const fn from_index(index: u8) -> Self {
        Self(index)
    }

    /// Creates a square from file (0–7) and rank (0–7).
    #[must_use]
    pub const fn from_file_rank(file: u8, rank: u8) -> Self {
        Self(rank * 8 + file)
    }

    /// Parses algebraic notation like `"e4"`.
    ///
    /// # Errors
    /// Returns `None` if the string is not a valid square name.
    #[must_use]
    pub fn from_algebraic(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let file = bytes[0].checked_sub(b'a').filter(|&f| f < 8)?;
        let rank = bytes[1].checked_sub(b'1').filter(|&r| r < 8)?;
        Some(Self::from_file_rank(file, rank))
    }

    /// Returns the 0–63 index of this square.
    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Returns the file (column) 0–7, where 0 = a-file.
    #[must_use]
    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    /// Returns the rank (row) 0–7, where 0 = rank 1.
    #[must_use]
    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    /// Returns the bit mask for this square in a `u64` bitboard.
    #[must_use]
    pub const fn bit(self) -> u64 {
        1u64 << self.0
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = b'a' + self.file();
        let rank = b'1' + self.rank();
        write!(f, "{}{}", file as char, rank as char)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_algebraic_e4() {
        let sq = Square::from_algebraic("e4").unwrap();
        assert_eq!(sq.index(), 28);
        assert_eq!(sq.file(), 4);
        assert_eq!(sq.rank(), 3);
    }

    #[test]
    fn from_algebraic_a1() {
        let sq = Square::from_algebraic("a1").unwrap();
        assert_eq!(sq.index(), 0);
    }

    #[test]
    fn from_algebraic_h8() {
        let sq = Square::from_algebraic("h8").unwrap();
        assert_eq!(sq.index(), 63);
    }

    #[test]
    fn display() {
        assert_eq!(Square::from_algebraic("e4").unwrap().to_string(), "e4");
        assert_eq!(Square::from_algebraic("a1").unwrap().to_string(), "a1");
        assert_eq!(Square::from_algebraic("h8").unwrap().to_string(), "h8");
    }

    #[test]
    fn invalid_square() {
        assert!(Square::from_algebraic("z9").is_none());
        assert!(Square::from_algebraic("e").is_none());
        assert!(Square::from_algebraic("").is_none());
    }
}
