//! Greedy self-play: model plays against itself to generate training data.
//!
//! No MCTS — each ply simply picks the move whose resulting position has the
//! highest value from the current player's perspective.  The terminal game
//! outcome labels every position in the game.

use crate::dataset::ChessDataset;
use crate::model::HybridValueNet;
use crate::pgn::{game_to_pgn, move_to_san, themed_filename};
use crate::position_db::fnv1a;
use chess::board::Board;
use chess::game::{game_status, game_status_with_history, GameStatus};
use chess::movegen::generate_legal_moves;
use chess::moves::Move;
use chess::piece::Color;
use tracing::{debug, info_span};

/// Plays `num_games` greedy games and collects all (position, outcome) pairs.
#[must_use]
pub fn generate(model: &HybridValueNet, num_games: usize) -> ChessDataset {
    let _span = info_span!("selfplay", total_games = num_games).entered();
    model.set_training(false);
    let mut dataset = ChessDataset::new();
    for game_idx in 0..num_games {
        let (samples, _moves, _outcome, _game_id) = play_game(model);
        let positions = samples.len();
        dataset.extend(samples);
        debug!(
            game = game_idx + 1,
            total = num_games,
            positions,
            "game complete"
        );
    }
    model.set_training(true);
    dataset
}

/// Plays `num_games` greedy games, collects training samples, and returns a
/// PGN string for every game alongside the dataset.
///
/// Each PGN string is paired with a unique themed filename (e.g.
/// `"dynamic-rook-3fa2.pgn"`).
#[must_use]
pub fn generate_with_pgn(
    model: &HybridValueNet,
    num_games: usize,
) -> (ChessDataset, Vec<(String, String)>) {
    let _span = info_span!("selfplay", total_games = num_games).entered();
    model.set_training(false);
    let mut dataset = ChessDataset::new();
    let mut pgns: Vec<(String, String)> = Vec::with_capacity(num_games);

    for game_idx in 0..num_games {
        let (samples, moves, outcome, game_id) = play_game(model);
        let positions = samples.len();
        dataset.extend(samples);

        let pgn = game_to_pgn(&moves, outcome, game_idx + 1);
        let filename = themed_filename(game_id);
        pgns.push((filename, pgn));

        debug!(
            game = game_idx + 1,
            total = num_games,
            positions,
            "game complete"
        );
    }
    model.set_training(true);
    (dataset, pgns)
}

/// Internal return type for a completed self-play game.
type PlayedGame = (Vec<(Board, f32, u64)>, Vec<(Board, Move)>, f32, u64);

/// Plays a single game.  Returns:
/// - training samples `(board_before_move, outcome, game_id)`
/// - the ordered `(board_before, move)` pairs for PGN serialisation
/// - the raw outcome value
/// - the stable game ID
fn play_game(model: &HybridValueNet) -> PlayedGame {
    let mut board = Board::starting_position();
    let mut history: Vec<Board> = Vec::new();
    let mut move_log: Vec<(Board, Move)> = Vec::new();
    let max_ply = 400; // prevent infinite games

    for _ in 0..max_ply {
        match game_status_with_history(&board, &history) {
            GameStatus::Ongoing => {}
            _ => break,
        }

        history.push(board.clone());

        let legal = generate_legal_moves(&board);
        if legal.is_empty() {
            break;
        }

        // Pick the move maximising value from the side-to-move's perspective.
        let best_move = legal.iter().copied().max_by(|&a, &b| {
            let va = eval_move(model, &board, a);
            let vb = eval_move(model, &board, b);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(mv) = best_move {
            let san = move_to_san(&board, mv);
            move_log.push((board.clone(), mv));
            board = board.make_move(mv);
            let turn = match board.side_to_move {
                Color::White => "White",
                Color::Black => "Black",
            };
            let eval = model.forward(&board).data()[0];
            debug!(mv = %san, turn, eval = format!("{eval:+.3}"), board = %format!("\n{board}"));
        } else {
            break;
        }
    }

    let outcome = terminal_outcome(&board);
    // Derive a stable game ID from the sequence of positions played.
    let game_id = fnv1a(history.iter().flat_map(|b| b.to_fen().into_bytes()));

    let samples = history.into_iter().map(|b| (b, outcome, game_id)).collect();

    (samples, move_log, outcome, game_id)
}

/// Evaluates a candidate move by running the model on the resulting position,
/// negating for Black (so higher is always better for the side to move).
fn eval_move(model: &HybridValueNet, board: &Board, mv: chess::moves::Move) -> f32 {
    let after = board.make_move(mv);
    let raw = model.forward(&after).data()[0];
    // White maximises positive values; Black maximises negative (flips sign).
    match board.side_to_move {
        Color::White => raw,
        Color::Black => -raw,
    }
}

/// Returns the game outcome from White's perspective after the position is terminal.
fn terminal_outcome(board: &Board) -> f32 {
    match game_status(board) {
        GameStatus::Checkmate => {
            // The side to move was checkmated, so the *other* side won.
            match board.side_to_move {
                Color::White => -1.0, // Black gave checkmate
                Color::Black => 1.0,  // White gave checkmate
            }
        }
        _ => 0.0, // stalemate, draw, or max-ply reached
    }
}
