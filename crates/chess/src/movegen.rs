use crate::attack;
use crate::bitboard::pop_lsb;
use crate::board::Board;
use crate::moves::Move;
use crate::piece::{Color, PieceKind};
use crate::square::Square;

/// Returns all legal moves from the given position.
#[must_use]
pub fn generate_legal_moves(board: &Board) -> Vec<Move> {
    generate_pseudo_legal(board)
        .into_iter()
        .filter(|&mv| {
            let after = board.make_move(mv);
            !after.is_in_check(board.side_to_move)
        })
        .collect()
}

fn generate_pseudo_legal(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(48);
    moves.extend(pawn_moves(board));
    moves.extend(knight_moves(board));
    moves.extend(bishop_moves(board));
    moves.extend(rook_moves(board));
    moves.extend(queen_moves(board));
    moves.extend(king_moves(board));
    moves
}

fn pawn_moves(board: &Board) -> Vec<Move> {
    let color = board.side_to_move;
    let ci = color as usize;
    let ei = color.opposite() as usize;
    let occupied = board.all_occupied();
    let enemy = board.pieces[ei].iter().fold(0, |a, &b| a | b);

    let (push_dir, start_rank, promo_rank): (i32, u8, u8) = match color {
        Color::White => (8, 1, 6),
        Color::Black => (-8, 6, 1),
    };

    let mut moves = Vec::new();
    let mut pawns = board.pieces[ci][PieceKind::Pawn.index()];

    while pawns != 0 {
        let from = pop_lsb(&mut pawns);
        let from_idx = i32::from(from.index());

        // Single push
        let push1_idx = from_idx + push_dir;
        if let Ok(push1_u8) = u8::try_from(push1_idx) {
            let push1 = Square::from_index(push1_u8);
            if occupied & push1.bit() == 0 {
                add_pawn_move(from, push1, promo_rank, &mut moves);

                // Double push from starting rank
                if from.rank() == start_rank {
                    let push2_idx = from_idx + push_dir * 2;
                    if let Ok(push2_u8) = u8::try_from(push2_idx) {
                        let push2 = Square::from_index(push2_u8);
                        if occupied & push2.bit() == 0 {
                            moves.push(Move::normal(from, push2));
                        }
                    }
                }
            }
        }

        // Captures (diagonal attacks that hit an enemy)
        let attacks = attack::pawn_attacks(color, from);
        let mut captures = attacks & enemy;
        while captures != 0 {
            let to = pop_lsb(&mut captures);
            add_pawn_move(from, to, promo_rank, &mut moves);
        }

        // En passant
        if let Some(ep_sq) = board.en_passant {
            if attacks & ep_sq.bit() != 0 {
                moves.push(Move::en_passant(from, ep_sq));
            }
        }
    }

    moves
}

fn add_pawn_move(from: Square, to: Square, promo_rank: u8, moves: &mut Vec<Move>) {
    if from.rank() == promo_rank {
        for kind in [
            PieceKind::Queen,
            PieceKind::Rook,
            PieceKind::Bishop,
            PieceKind::Knight,
        ] {
            moves.push(Move::promotion(from, to, kind));
        }
    } else {
        moves.push(Move::normal(from, to));
    }
}

fn knight_moves(board: &Board) -> Vec<Move> {
    let color = board.side_to_move;
    let ci = color as usize;
    let own = board.pieces[ci].iter().fold(0, |a, &b| a | b);
    let mut moves = Vec::new();
    let mut knights = board.pieces[ci][PieceKind::Knight.index()];

    while knights != 0 {
        let from = pop_lsb(&mut knights);
        let mut targets = attack::knight_attacks(from) & !own;
        while targets != 0 {
            let to = pop_lsb(&mut targets);
            moves.push(Move::normal(from, to));
        }
    }
    moves
}

fn bishop_moves(board: &Board) -> Vec<Move> {
    sliding_piece_moves(board, PieceKind::Bishop, |sq, occ| {
        attack::bishop_attacks(sq, occ)
    })
}

fn rook_moves(board: &Board) -> Vec<Move> {
    sliding_piece_moves(board, PieceKind::Rook, |sq, occ| {
        attack::rook_attacks(sq, occ)
    })
}

fn queen_moves(board: &Board) -> Vec<Move> {
    sliding_piece_moves(board, PieceKind::Queen, |sq, occ| {
        attack::queen_attacks(sq, occ)
    })
}

fn sliding_piece_moves(
    board: &Board,
    kind: PieceKind,
    attack_fn: impl Fn(Square, u64) -> u64,
) -> Vec<Move> {
    let color = board.side_to_move;
    let ci = color as usize;
    let own = board.pieces[ci].iter().fold(0, |a, &b| a | b);
    let occupied = board.all_occupied();
    let mut moves = Vec::new();
    let mut pieces = board.pieces[ci][kind.index()];

    while pieces != 0 {
        let from = pop_lsb(&mut pieces);
        let mut targets = attack_fn(from, occupied) & !own;
        while targets != 0 {
            let to = pop_lsb(&mut targets);
            moves.push(Move::normal(from, to));
        }
    }
    moves
}

fn king_moves(board: &Board) -> Vec<Move> {
    let color = board.side_to_move;
    let ci = color as usize;
    let own = board.pieces[ci].iter().fold(0, |a, &b| a | b);
    let mut moves = Vec::new();

    let king_bb = board.pieces[ci][PieceKind::King.index()];
    if king_bb == 0 {
        return moves;
    }
    let from = crate::bitboard::lsb_square(king_bb);

    // Normal king moves
    let mut targets = attack::king_attacks(from) & !own;
    while targets != 0 {
        let to = pop_lsb(&mut targets);
        moves.push(Move::normal(from, to));
    }

    // Castling
    moves.extend(castling_moves(board, color, from));
    moves
}

fn castling_moves(board: &Board, color: Color, king_sq: Square) -> Vec<Move> {
    use crate::board::{BK, BQ, WK, WQ};

    let mut moves = Vec::new();
    let occupied = board.all_occupied();

    // We compute enemy attacks lazily — only if castling rights are set
    let in_check = board.is_in_check(color);
    if in_check {
        return moves; // Can't castle while in check
    }

    let enemy_attacks = || attack::all_attacks(color.opposite(), &board.pieces, occupied);

    match color {
        Color::White => {
            // Kingside: e1-f1-g1 must be empty and f1/g1 not attacked
            if board.castling & WK != 0 {
                let f1 = Square::from_algebraic("f1").unwrap();
                let g1 = Square::from_algebraic("g1").unwrap();
                if occupied & (f1.bit() | g1.bit()) == 0 {
                    let ea = enemy_attacks();
                    if ea & (f1.bit() | g1.bit()) == 0 {
                        moves.push(Move::castling(king_sq, g1));
                    }
                }
            }
            // Queenside: e1-d1-c1 must be unattacked, b1-c1-d1 unoccupied
            if board.castling & WQ != 0 {
                let b1 = Square::from_algebraic("b1").unwrap();
                let c1 = Square::from_algebraic("c1").unwrap();
                let d1 = Square::from_algebraic("d1").unwrap();
                if occupied & (b1.bit() | c1.bit() | d1.bit()) == 0 {
                    let ea = enemy_attacks();
                    if ea & (c1.bit() | d1.bit()) == 0 {
                        moves.push(Move::castling(king_sq, c1));
                    }
                }
            }
        }
        Color::Black => {
            // Kingside
            if board.castling & BK != 0 {
                let f8 = Square::from_algebraic("f8").unwrap();
                let g8 = Square::from_algebraic("g8").unwrap();
                if occupied & (f8.bit() | g8.bit()) == 0 {
                    let ea = enemy_attacks();
                    if ea & (f8.bit() | g8.bit()) == 0 {
                        moves.push(Move::castling(king_sq, g8));
                    }
                }
            }
            // Queenside
            if board.castling & BQ != 0 {
                let b8 = Square::from_algebraic("b8").unwrap();
                let c8 = Square::from_algebraic("c8").unwrap();
                let d8 = Square::from_algebraic("d8").unwrap();
                if occupied & (b8.bit() | c8.bit() | d8.bit()) == 0 {
                    let ea = enemy_attacks();
                    if ea & (c8.bit() | d8.bit()) == 0 {
                        moves.push(Move::castling(king_sq, c8));
                    }
                }
            }
        }
    }
    moves
}

/// Counts all nodes at exactly `depth` from the given position (perft).
/// Used for correctness testing of move generation.
#[must_use]
pub fn perft(board: &Board, depth: u32) -> u64 {
    let _span = tracing::debug_span!("perft", depth).entered();
    if depth == 0 {
        return 1;
    }
    let moves = generate_legal_moves(board);
    if depth == 1 {
        return moves.len() as u64;
    }
    moves
        .iter()
        .map(|&mv| perft(&board.make_move(mv), depth - 1))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::from_fen;
    use crate::moves::MoveKind;

    fn board(fen: &str) -> Board {
        from_fen(fen).unwrap()
    }

    #[test]
    fn starting_position_20_moves() {
        let b = Board::starting_position();
        assert_eq!(generate_legal_moves(&b).len(), 20);
    }

    #[test]
    fn perft_depth_1() {
        assert_eq!(perft(&Board::starting_position(), 1), 20);
    }

    #[test]
    fn perft_depth_2() {
        assert_eq!(perft(&Board::starting_position(), 2), 400);
    }

    #[test]
    fn perft_depth_3() {
        assert_eq!(perft(&Board::starting_position(), 3), 8902);
    }

    #[test]
    fn en_passant_included() {
        let b = board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
        let moves = generate_legal_moves(&b);
        assert!(
            moves
                .iter()
                .any(|m| m.kind == MoveKind::EnPassant
                    && m.to == Square::from_algebraic("f6").unwrap()),
            "en passant to f6 should be legal"
        );
    }

    #[test]
    fn castling_both_sides() {
        let b = board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        let moves = generate_legal_moves(&b);
        let castles: Vec<_> = moves
            .iter()
            .filter(|m| m.kind == MoveKind::Castling)
            .collect();
        assert_eq!(castles.len(), 2, "white should have both castling moves");
    }

    #[test]
    fn promotion_has_four_moves() {
        let b = board("8/P7/8/8/8/8/8/4K2k w - - 0 1");
        let moves = generate_legal_moves(&b);
        let promos: Vec<_> = moves.iter().filter(|m| m.promotion.is_some()).collect();
        assert_eq!(promos.len(), 4, "should generate Q/R/B/N promotions");
    }

    #[test]
    fn pinned_rook_cannot_leave_file() {
        // White rook on e2 is pinned by black rook on e4, white king on e1
        let b = board("4k3/8/8/8/4r3/8/4R3/4K3 w - - 0 1");
        let moves = generate_legal_moves(&b);
        // The pinned rook on e2 can only move along the e-file
        let rook_moves: Vec<_> = moves
            .iter()
            .filter(|m| m.from == Square::from_algebraic("e2").unwrap())
            .collect();
        for m in &rook_moves {
            assert_eq!(m.to.file(), 4, "pinned rook must stay on e-file");
        }
    }
}
