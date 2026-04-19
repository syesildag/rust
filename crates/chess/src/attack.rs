//! Pre-computed attack tables and ray-casting for all piece types.
//!
//! ## Tables
//!
//! Attack tables are initialised once at first use via [`std::sync::OnceLock`] and
//! stored in static memory. There are four table types:
//!
//! - **Knight** — 64 precomputed bitboards, one per square.
//! - **King** — 64 precomputed bitboards, one per square.
//! - **Pawn** — 2 × 64 precomputed bitboards (one set per color).
//! - **Ray** — 64 × 8 precomputed bitboards. Each entry is the set of squares
//!   reachable from a given square in one of 8 directions, excluding the origin.
//!
//! ## Sliding piece attacks (ray casting)
//!
//! The 8 directions are indexed: N=0, NE=1, E=2, NW=3, S=4, SW=5, W=6, SE=7.
//!
//! For each ray direction from a given square:
//! 1. Intersect the precomputed ray bitboard with the occupied squares.
//! 2. Find the first blocker along the ray (LSB for positive rays, MSB for negative).
//! 3. Mask out everything beyond that blocker — the blocker's own square is included
//!    (it can be captured).
//!
//! Using precomputed rays makes each direction O(1). Rook attacks = N+S+E+W rays;
//! bishop = NE+NW+SE+SW.

use std::sync::OnceLock;

use crate::piece::Color;
use crate::square::Square;

// Positive rays (bit index increases along the ray): N=0, NE=1, E=2, NW=3
// Negative rays (bit index decreases along the ray): S=4, SW=5, W=6, SE=7
pub const N: usize = 0;
pub const NE: usize = 1;
pub const E: usize = 2;
pub const NW: usize = 3;
pub const S: usize = 4;
pub const SW: usize = 5;
pub const W: usize = 6;
pub const SE: usize = 7;

struct Tables {
    knight: [u64; 64],
    king: [u64; 64],
    pawn: [[u64; 64]; 2],
    ray: [[u64; 8]; 64],
}

static TABLES: OnceLock<Tables> = OnceLock::new();

fn get_tables() -> &'static Tables {
    TABLES.get_or_init(init_tables)
}

fn init_tables() -> Tables {
    let mut knight = [0u64; 64];
    let mut king = [0u64; 64];
    let mut pawn = [[0u64; 64]; 2];
    let mut ray = [[0u64; 8]; 64];

    for sq in 0u8..64 {
        let bb = 1u64 << sq;
        let file = i32::from(sq % 8);

        // Knight attacks
        let nk = &mut knight[sq as usize];
        if file > 1 {
            *nk |= (bb << 6) | (bb >> 10);
        }
        if file > 0 {
            *nk |= (bb << 15) | (bb >> 17);
        }
        if file < 7 {
            *nk |= (bb << 17) | (bb >> 15);
        }
        if file < 6 {
            *nk |= (bb << 10) | (bb >> 6);
        }

        // King attacks
        let kg = &mut king[sq as usize];
        if file > 0 {
            *kg |= (bb >> 1) | (bb << 7) | (bb >> 9);
        }
        if file < 7 {
            *kg |= (bb << 1) | (bb << 9) | (bb >> 7);
        }
        *kg |= (bb << 8) | (bb >> 8);

        // Pawn attacks
        if file > 0 {
            pawn[Color::White as usize][sq as usize] |= bb << 7;
            pawn[Color::Black as usize][sq as usize] |= bb >> 9;
        }
        if file < 7 {
            pawn[Color::White as usize][sq as usize] |= bb << 9;
            pawn[Color::Black as usize][sq as usize] |= bb >> 7;
        }

        ray[sq as usize] = compute_rays(sq);
    }

    Tables {
        knight,
        king,
        pawn,
        ray,
    }
}

fn compute_rays(sq: u8) -> [u64; 8] {
    let mut rays = [0u64; 8];
    let file = sq % 8;
    let rank = sq / 8;

    // N (positive, +8)
    for r in (rank + 1)..8 {
        rays[N] |= 1u64 << (r * 8 + file);
    }
    // NE (positive, +9)
    {
        let (mut r, mut f) = (rank + 1, file + 1);
        while r < 8 && f < 8 {
            rays[NE] |= 1u64 << (r * 8 + f);
            r += 1;
            f += 1;
        }
    }
    // E (positive, +1)
    for f in (file + 1)..8 {
        rays[E] |= 1u64 << (rank * 8 + f);
    }
    // NW (positive, +7)
    {
        let mut r = rank + 1;
        let mut f = file;
        while r < 8 && f > 0 {
            f -= 1;
            rays[NW] |= 1u64 << (r * 8 + f);
            r += 1;
        }
    }
    // S (negative, -8)
    for r in 0..rank {
        rays[S] |= 1u64 << (r * 8 + file);
    }
    // SW (negative, -9)
    {
        let mut r = rank;
        let mut f = file;
        while r > 0 && f > 0 {
            r -= 1;
            f -= 1;
            rays[SW] |= 1u64 << (r * 8 + f);
        }
    }
    // W (negative, -1)
    for f in 0..file {
        rays[W] |= 1u64 << (rank * 8 + f);
    }
    // SE (negative, -7)
    {
        let mut r = rank;
        let mut f = file + 1;
        while r > 0 && f < 8 {
            r -= 1;
            rays[SE] |= 1u64 << (r * 8 + f);
            f += 1;
        }
    }

    rays
}

/// o^(o-2r) for a positive ray (bit index increases along ray).
fn positive_ray_attacks(ray: u64, occupied: u64, sq_bit: u64) -> u64 {
    let o = occupied & ray;
    (o.wrapping_sub(sq_bit.wrapping_shl(1)) ^ o) & ray
}

/// Bit-reversed o^(o-2r) for a negative ray (bit index decreases along ray).
fn negative_ray_attacks(ray: u64, occupied: u64, sq_bit: u64) -> u64 {
    let rev_o = (occupied & ray).reverse_bits();
    let rev_r = sq_bit.reverse_bits();
    (rev_o.wrapping_sub(rev_r.wrapping_shl(1)) ^ rev_o).reverse_bits() & ray
}

/// Attacks from a sliding piece on `sq` with `occupied` squares, for the given direction indices.
#[must_use]
pub fn sliding_attacks(sq: Square, occupied: u64, dirs: &[usize]) -> u64 {
    let tables = get_tables();
    let sq_bit = sq.bit();
    let mut attacks = 0u64;
    for &dir in dirs {
        let ray = tables.ray[sq.index() as usize][dir];
        attacks |= if dir < 4 {
            positive_ray_attacks(ray, occupied, sq_bit)
        } else {
            negative_ray_attacks(ray, occupied, sq_bit)
        };
    }
    attacks
}

/// Returns the bitboard of squares a rook on `sq` can reach given `occupied` squares.
///
/// Casts rays in the N, E, S, and W directions; blockers are included (capturable).
#[must_use]
pub fn rook_attacks(sq: Square, occupied: u64) -> u64 {
    sliding_attacks(sq, occupied, &[N, E, S, W])
}

/// Returns the bitboard of squares a bishop on `sq` can reach given `occupied` squares.
///
/// Casts rays in the NE, NW, SW, and SE directions; blockers are included (capturable).
#[must_use]
pub fn bishop_attacks(sq: Square, occupied: u64) -> u64 {
    sliding_attacks(sq, occupied, &[NE, NW, SW, SE])
}

/// Returns the bitboard of squares a queen on `sq` can reach given `occupied` squares.
///
/// Equivalent to `rook_attacks | bishop_attacks` for the same square.
#[must_use]
pub fn queen_attacks(sq: Square, occupied: u64) -> u64 {
    rook_attacks(sq, occupied) | bishop_attacks(sq, occupied)
}

/// Returns the precomputed bitboard of squares a knight on `sq` can jump to.
///
/// The result is independent of occupied squares because knights leap over pieces.
#[must_use]
pub fn knight_attacks(sq: Square) -> u64 {
    get_tables().knight[sq.index() as usize]
}

/// Returns the precomputed bitboard of squares a king on `sq` can move to (one step in any direction).
///
/// The result is independent of occupied squares; legality filtering is done by the caller.
#[must_use]
pub fn king_attacks(sq: Square) -> u64 {
    get_tables().king[sq.index() as usize]
}

/// Returns the bitboard of squares attacked diagonally by a pawn of `color` on `sq`.
///
/// White pawns attack toward higher ranks; black pawns attack toward lower ranks.
/// This does **not** include forward pushes, only diagonal captures.
#[must_use]
pub fn pawn_attacks(color: Color, sq: Square) -> u64 {
    get_tables().pawn[color as usize][sq.index() as usize]
}

/// Returns the union of all squares attacked by every piece of `color`.
///
/// Used to detect check and to validate that castling transit squares are not under attack.
#[must_use]
pub fn all_attacks(color: Color, pieces: &[[u64; 6]; 2], occupied: u64) -> u64 {
    let ci = color as usize;
    let mut attacks = 0u64;

    let mut pawns = pieces[ci][crate::piece::PieceKind::Pawn.index()];
    while pawns != 0 {
        let sq = crate::bitboard::pop_lsb(&mut pawns);
        attacks |= pawn_attacks(color, sq);
    }
    let mut knights = pieces[ci][crate::piece::PieceKind::Knight.index()];
    while knights != 0 {
        let sq = crate::bitboard::pop_lsb(&mut knights);
        attacks |= knight_attacks(sq);
    }
    let mut bishops = pieces[ci][crate::piece::PieceKind::Bishop.index()];
    while bishops != 0 {
        let sq = crate::bitboard::pop_lsb(&mut bishops);
        attacks |= bishop_attacks(sq, occupied);
    }
    let mut rooks = pieces[ci][crate::piece::PieceKind::Rook.index()];
    while rooks != 0 {
        let sq = crate::bitboard::pop_lsb(&mut rooks);
        attacks |= rook_attacks(sq, occupied);
    }
    let mut queens = pieces[ci][crate::piece::PieceKind::Queen.index()];
    while queens != 0 {
        let sq = crate::bitboard::pop_lsb(&mut queens);
        attacks |= queen_attacks(sq, occupied);
    }
    let king = pieces[ci][crate::piece::PieceKind::King.index()];
    if king != 0 {
        attacks |= king_attacks(crate::bitboard::lsb_square(king));
    }

    attacks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(s: &str) -> Square {
        Square::from_algebraic(s).unwrap()
    }

    #[test]
    fn knight_on_a1_has_two_attacks() {
        let attacks = knight_attacks(sq("a1"));
        assert_eq!(attacks.count_ones(), 2);
        assert!(attacks & sq("b3").bit() != 0);
        assert!(attacks & sq("c2").bit() != 0);
    }

    #[test]
    fn knight_on_d4_has_eight_attacks() {
        assert_eq!(knight_attacks(sq("d4")).count_ones(), 8);
    }

    #[test]
    fn king_on_e1_has_five_attacks() {
        assert_eq!(king_attacks(sq("e1")).count_ones(), 5);
    }

    #[test]
    fn rook_open_board() {
        // Rook on e4, empty board: 7 along rank + 7 along file = 14
        assert_eq!(rook_attacks(sq("e4"), 0).count_ones(), 14);
    }

    #[test]
    fn rook_blocked() {
        let occupied = sq("e6").bit() | sq("e2").bit();
        let attacks = rook_attacks(sq("e4"), occupied);
        assert!(attacks & sq("e6").bit() != 0, "can capture e6 blocker");
        assert!(attacks & sq("e7").bit() == 0, "blocked beyond e6");
        assert!(attacks & sq("e2").bit() != 0, "can capture e2 blocker");
        assert!(attacks & sq("e1").bit() == 0, "blocked beyond e2");
    }

    #[test]
    fn bishop_open_board() {
        assert_eq!(bishop_attacks(sq("e4"), 0).count_ones(), 13);
    }

    #[test]
    fn bishop_blocked() {
        let occupied = sq("g6").bit();
        let attacks = bishop_attacks(sq("e4"), occupied);
        assert!(attacks & sq("g6").bit() != 0, "can capture g6");
        assert!(attacks & sq("h7").bit() == 0, "blocked beyond g6");
    }

    #[test]
    fn pawn_attacks_white() {
        let attacks = pawn_attacks(Color::White, sq("e4"));
        assert!(attacks & sq("d5").bit() != 0);
        assert!(attacks & sq("f5").bit() != 0);
        assert_eq!(attacks.count_ones(), 2);
    }

    #[test]
    fn pawn_attacks_black() {
        let attacks = pawn_attacks(Color::Black, sq("e5"));
        assert!(attacks & sq("d4").bit() != 0);
        assert!(attacks & sq("f4").bit() != 0);
    }
}
