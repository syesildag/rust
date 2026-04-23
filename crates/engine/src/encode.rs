//! Board-to-tensor encoding for the neural network.
//!
//! ## 18-Channel Plane Layout
//!
//! A board position is encoded as a `[18, 8, 8]` float32 tensor.
//! Each of the 18 channels is a binary 8×8 plane:
//!
//! | Channel | Content |
//! |---------|---------|
//! | 0–5     | White pieces: Pawn, Knight, Bishop, Rook, Queen, King |
//! | 6–11    | Black pieces: Pawn, Knight, Bishop, Rook, Queen, King |
//! | 12      | Side to move (all 1.0 = White to move, all 0.0 = Black) |
//! | 13      | White kingside castling right |
//! | 14      | White queenside castling right |
//! | 15      | Black kingside castling right |
//! | 16      | Black queenside castling right |
//! | 17      | En passant target square (1.0 at the target square, all 0.0 if none) |
//!
//! ## Design rationale
//!
//! - **Piece planes (0–11):** Mirror the `Board::pieces` bitboard layout. Each
//!   square is 1.0 if that piece occupies it, 0.0 otherwise.
//! - **Side-to-move plane (12):** Gives the network an explicit global signal
//!   about whose turn it is without requiring inference from piece positions.
//! - **Castling planes (13–16):** Constant planes (all 1.0 or all 0.0) that
//!   communicate castling availability as a simple spatial mask.
//! - **En passant plane (17):** A point plane with a single 1.0 at the target
//!   square a pawn may capture to, following the AlphaZero convention.  All
//!   zeros when no en passant capture is available.
//!
//! This encoding follows the AlphaZero convention and is designed for 2-D
//! convolutional layers that treat the 8×8 board as a spatial grid.

use chess::board::{Board, BK, BQ, WK, WQ};
use chess::piece::Color;
use chess::square::Square;
use tensor::Tensor;

/// Encodes a single board position into a `[18, 8, 8]` float32 tensor.
///
/// See the module-level documentation for the full channel layout.
#[must_use]
pub fn encode(board: &Board) -> Tensor {
    let mut data = vec![0.0f32; 18 * 8 * 8];

    // Piece planes 0–11.
    for sq_idx in 0u8..64 {
        let sq = Square::from_index(sq_idx);
        if let Some(piece) = board.piece_at(sq) {
            let color_offset = match piece.color {
                Color::White => 0,
                Color::Black => 6,
            };
            let plane = color_offset + piece.kind.index();
            let rank = sq.rank() as usize;
            let file = sq.file() as usize;
            data[plane * 64 + rank * 8 + file] = 1.0;
        }
    }

    // Plane 12: side to move.
    let stm_val = if board.side_to_move == Color::White {
        1.0
    } else {
        0.0
    };
    for i in 0..64 {
        data[12 * 64 + i] = stm_val;
    }

    // Planes 13–16: castling rights.
    let castling_flags = [(WK, 13usize), (WQ, 14), (BK, 15), (BQ, 16)];
    for (flag, plane) in castling_flags {
        if board.castling & flag != 0 {
            for i in 0..64 {
                data[plane * 64 + i] = 1.0;
            }
        }
    }

    // Plane 17: en passant target square.
    if let Some(ep) = board.en_passant {
        let rank = ep.rank() as usize;
        let file = ep.file() as usize;
        data[17 * 64 + rank * 8 + file] = 1.0;
    }

    Tensor::from_vec(data, &[18, 8, 8])
}

/// Encodes a board position as a batch tensor of shape `[1, 18, 8, 8]`.
///
/// Equivalent to wrapping `encode(board)` in a batch dimension.
/// This is the format expected by `HybridValueNet::forward`.
#[must_use]
pub fn encode_batch(board: &Board) -> Tensor {
    encode(board).reshape(&[1, 18, 8, 8])
}

/// Encodes a slice of board positions into a single `[B, 18, 8, 8]` batch tensor.
#[must_use]
pub fn encode_boards(boards: &[Board]) -> Tensor {
    let b = boards.len();
    let mut data = Vec::with_capacity(b * 18 * 64);
    for board in boards {
        data.extend_from_slice(&encode(board).data());
    }
    Tensor::from_vec(data, &[b, 18, 8, 8])
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use chess::piece::PieceKind;

    #[test]
    fn starting_position_planes() {
        let board = Board::starting_position();
        let t = encode(&board);
        assert_eq!(t.shape(), &[18, 8, 8]);
        let d = t.data();
        // White pawns on rank 2 (plane 0): 8 ones.
        let white_pawn_plane: Vec<f32> = d[0..64].to_vec();
        assert_eq!(white_pawn_plane.iter().filter(|&&v| v == 1.0).count(), 8);
        // Side-to-move plane (12) should be all ones (White to move).
        let stm: Vec<f32> = d[12 * 64..13 * 64].to_vec();
        assert!(stm.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn piece_kind_ordering() {
        // PieceKind::ALL order: Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5
        assert_eq!(PieceKind::Pawn.index(), 0);
        assert_eq!(PieceKind::King.index(), 5);
    }

    #[test]
    fn en_passant_plane() {
        use chess::fen;
        // After 1.e4 e5 2.d4: en passant target is d3 (file=3, rank=2).
        let board = fen::from_fen("rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2")
            .expect("valid FEN");
        let t = encode(&board);
        let d = t.data();
        // Plane 17 should have exactly one 1.0 at d3 (rank=2, file=3).
        let ep_plane: Vec<f32> = d[17 * 64..18 * 64].to_vec();
        let ones: Vec<usize> = ep_plane
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == 1.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(ones.len(), 1);
        // index = rank * 8 + file = 2 * 8 + 3 = 19
        assert_eq!(ones[0], 19);
    }

    #[test]
    fn no_en_passant_plane_all_zeros() {
        let board = chess::board::Board::starting_position();
        let t = encode(&board);
        let d = t.data();
        let ep_plane: Vec<f32> = d[17 * 64..18 * 64].to_vec();
        assert!(ep_plane.iter().all(|&v| v == 0.0));
    }
}
