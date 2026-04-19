use std::fmt;

use crate::square::Square;

/// A 64-bit integer representing a set of squares; bit `n` corresponds to square index `n` (a1=bit 0, h8=bit 63).
pub type Bitboard = u64;

/// Returns a bitboard with the given square's bit set.
#[must_use]
pub const fn set_bit(bb: Bitboard, sq: Square) -> Bitboard {
    bb | sq.bit()
}

/// Returns a bitboard with the given square's bit cleared.
#[must_use]
pub const fn clear_bit(bb: Bitboard, sq: Square) -> Bitboard {
    bb & !sq.bit()
}

/// Returns the square of the least-significant set bit.
///
/// # Panics
/// Calling with `bb == 0` produces an out-of-range square index; callers must ensure `bb != 0`.
#[must_use]
pub fn lsb_square(bb: Bitboard) -> Square {
    #[allow(clippy::cast_possible_truncation)]
    Square::from_index(bb.trailing_zeros() as u8)
}

/// Clears the least-significant set bit of `*bb` in place and returns its square.
///
/// # Panics
/// Calling with `*bb == 0` produces an out-of-range square index; callers must ensure `*bb != 0`.
#[must_use]
pub fn pop_lsb(bb: &mut Bitboard) -> Square {
    let sq = lsb_square(*bb);
    *bb &= *bb - 1;
    sq
}

/// Returns the number of set bits (pieces) in a bitboard.
#[must_use]
pub const fn count_bits(bb: Bitboard) -> u32 {
    bb.count_ones()
}

/// Prints a bitboard as an 8×8 grid for debugging (rank 8 at top).
pub struct BitboardDisplay(pub Bitboard);

impl fmt::Display for BitboardDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = Square::from_file_rank(file, rank);
                if self.0 & sq.bit() != 0 {
                    write!(f, "1 ")?;
                } else {
                    write!(f, ". ")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_clear() {
        let sq = Square::from_algebraic("e4").unwrap();
        let bb = set_bit(0, sq);
        assert_eq!(bb, 1u64 << 28);
        assert_eq!(clear_bit(bb, sq), 0);
    }

    #[test]
    fn lsb_and_pop() {
        let a1 = Square::from_algebraic("a1").unwrap();
        let e4 = Square::from_algebraic("e4").unwrap();
        let mut bb = a1.bit() | e4.bit();
        let first = pop_lsb(&mut bb);
        assert_eq!(first.index(), a1.index());
        assert_eq!(count_bits(bb), 1);
    }

    #[test]
    fn count_bits_starting_position() {
        // 16 pawns + 4 rooks + 4 knights + 4 bishops + 2 queens + 2 kings = 32
        let all_pieces: u64 = 0xFFFF_0000_0000_FFFF;
        assert_eq!(count_bits(all_pieces), 32);
    }
}
