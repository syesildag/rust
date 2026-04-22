use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::piece::{Color, PieceKind};

/// Reason a game ended in a draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawReason {
    /// Neither side has made a pawn move or capture in the last 50 full moves (100 half-moves).
    FiftyMoveRule,
    /// Neither side has enough material to deliver checkmate (kings only, or with one minor piece).
    InsufficientMaterial,
    /// The same position has occurred three times.
    ThreefoldRepetition,
}

/// The current status of the game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameStatus {
    /// The side to move has at least one legal move.
    Ongoing,
    /// The side to move has no legal moves and their king is in check.
    Checkmate,
    /// The side to move has no legal moves and their king is not in check.
    Stalemate,
    /// The game is drawn for the given reason.
    Draw(DrawReason),
}

/// Returns `true` when `a` and `b` represent the same chess position for the
/// purposes of repetition detection (pieces, side to move, castling rights, and
/// en-passant square; ignores clocks).
fn same_position(a: &Board, b: &Board) -> bool {
    a.pieces == b.pieces
        && a.side_to_move == b.side_to_move
        && a.castling == b.castling
        && a.en_passant == b.en_passant
}

/// Like [`game_status`] but also detects threefold repetition.
///
/// `history` should contain the sequence of board positions that were reached
/// *before* each move was played (i.e. the positions already seen).  If the
/// current position matches two or more entries in `history` the function
/// returns [`GameStatus::Draw(DrawReason::ThreefoldRepetition)`].
#[must_use]
pub fn game_status_with_history(board: &Board, history: &[Board]) -> GameStatus {
    let repetitions = history.iter().filter(|b| same_position(b, board)).count();
    if repetitions >= 2 {
        return GameStatus::Draw(DrawReason::ThreefoldRepetition);
    }
    game_status(board)
}

/// Determines the current game status for the side to move.
///
/// Note: threefold repetition is **not** detected; callers that need it must
/// track position history themselves.  Use [`game_status_with_history`] when a
/// history is available.
#[must_use]
pub fn game_status(board: &Board) -> GameStatus {
    if board.halfmove_clock >= 100 {
        return GameStatus::Draw(DrawReason::FiftyMoveRule);
    }
    if has_insufficient_material(board) {
        return GameStatus::Draw(DrawReason::InsufficientMaterial);
    }
    let legal = generate_legal_moves(board);
    if legal.is_empty() {
        if board.is_in_check(board.side_to_move) {
            GameStatus::Checkmate
        } else {
            GameStatus::Stalemate
        }
    } else {
        GameStatus::Ongoing
    }
}

/// Returns `true` when neither side has sufficient mating material.
///
/// Recognized draw patterns:
/// - Kings only
/// - King + bishop vs king
/// - King + knight vs king
fn has_insufficient_material(board: &Board) -> bool {
    let white_count = board.white_occupied().count_ones();
    let black_count = board.black_occupied().count_ones();

    // More than 3 pieces total → sufficient material may exist
    if white_count + black_count > 3 {
        return false;
    }

    // Both kings only
    if white_count == 1 && black_count == 1 {
        return true;
    }

    // One side has exactly one minor piece (bishop or knight)
    for color in [Color::White, Color::Black] {
        let ci = color as usize;
        let own_count = if color == Color::White {
            white_count
        } else {
            black_count
        };
        let opp_count = if color == Color::White {
            black_count
        } else {
            white_count
        };
        if own_count == 2 && opp_count == 1 {
            let bishops = board.pieces[ci][PieceKind::Bishop.index()].count_ones();
            let knights = board.pieces[ci][PieceKind::Knight.index()].count_ones();
            if bishops == 1 || knights == 1 {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::from_fen;

    fn board(fen: &str) -> Board {
        from_fen(fen).unwrap()
    }

    #[test]
    fn ongoing_at_start() {
        assert_eq!(
            game_status(&Board::starting_position()),
            GameStatus::Ongoing
        );
    }

    #[test]
    fn scholars_mate_is_checkmate() {
        // Position after Scholar's mate — white is checkmated
        let b = board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
        assert_eq!(game_status(&b), GameStatus::Checkmate);
    }

    #[test]
    fn stalemate_position() {
        let b = board("k7/8/1Q6/8/8/8/8/7K b - - 0 1");
        assert_eq!(game_status(&b), GameStatus::Stalemate);
    }

    #[test]
    fn fifty_move_draw() {
        // Modify starting position halfmove clock to 100
        let mut b = Board::starting_position();
        b.halfmove_clock = 100;
        assert_eq!(game_status(&b), GameStatus::Draw(DrawReason::FiftyMoveRule));
    }

    #[test]
    fn insufficient_material_kings_only() {
        let b = board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        assert_eq!(
            game_status(&b),
            GameStatus::Draw(DrawReason::InsufficientMaterial)
        );
    }

    #[test]
    fn insufficient_material_king_knight() {
        let b = board("4k3/8/8/8/8/8/8/4KN2 w - - 0 1");
        assert_eq!(
            game_status(&b),
            GameStatus::Draw(DrawReason::InsufficientMaterial)
        );
    }

    #[test]
    fn threefold_repetition_draw() {
        let b = Board::starting_position();
        // Two prior occurrences of the same position in history → draw on third.
        let history = vec![b.clone(), b.clone()];
        assert_eq!(
            game_status_with_history(&b, &history),
            GameStatus::Draw(DrawReason::ThreefoldRepetition)
        );
    }

    #[test]
    fn no_repetition_without_history() {
        let b = Board::starting_position();
        let history = vec![b.clone()]; // only one prior occurrence
        assert_eq!(game_status_with_history(&b, &history), GameStatus::Ongoing);
    }
}
