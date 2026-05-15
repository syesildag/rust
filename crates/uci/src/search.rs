use chess::board::Board;
use chess::game::{game_status, GameStatus};
use chess::movegen::generate_legal_moves;
use chess::moves::Move;
use chess::piece::{Color, PieceKind};
use chess::square::Square;
use engine::model::HybridValueNet;
use tracing::info;

pub const SEARCH_DEPTH: u32 = 1;

/// Maximum boards per `forward_batch` call.
/// Tightest Metal constraint: the attention softmax dispatches B×8×65 = B×520
/// workgroups; Metal's per-dimension limit is 65,535, so B ≤ 126.
/// 120 gives a ~5% safety margin.
const BATCH_CHUNK: usize = 120;

/// Returns the best move and its evaluation (current player's perspective, +1 = winning).
/// Uses batch minimax: all leaf positions are collected first and evaluated with
/// chunked `forward_batch()` calls, avoiding per-leaf GPU dispatch overhead.
pub fn best_move(model: &HybridValueNet, board: &Board, depth: u32) -> Option<(Move, f32)> {
    let legal = generate_legal_moves(board);
    if legal.is_empty() {
        return None;
    }
    let sign = match board.side_to_move {
        Color::White => 1.0_f32,
        Color::Black => -1.0_f32,
    };

    // Collect every leaf board reachable in `depth` plies.
    let mut leaves: Vec<Board> = Vec::new();
    collect_leaves(board, depth, &mut leaves);

    info!(depth, moves = legal.len(), leaves = leaves.len(), "search start");

    // Evaluate leaves in chunks to stay within the GPU 256 MB buffer limit.
    let scores: Vec<f32> = leaves
        .chunks(BATCH_CHUNK)
        .flat_map(|chunk| model.forward_batch(chunk).data().to_vec())
        .collect();

    info!(leaves = scores.len(), "batch eval done");

    // Propagate minimax values back to the root children.
    let mut idx = 0usize;
    let mut best: Option<(usize, f32)> = None;

    for (i, &mv) in legal.iter().enumerate() {
        let child = board.make_move(mv);
        let v = minimax_node(&child, depth.saturating_sub(1), &scores, &mut idx);
        if best.map_or(true, |(_, bv)| sign * v > sign * bv) {
            best = Some((i, v));
        }
    }

    best.map(|(i, v)| (legal[i], sign * v))
}

/// Returns the game-theoretic value from White's perspective for terminal positions.
fn terminal_value(board: &Board) -> f32 {
    match game_status(board) {
        GameStatus::Checkmate => match board.side_to_move {
            Color::White => -1.0, // White is mated
            Color::Black => 1.0,  // Black is mated
        },
        _ => 0.0,
    }
}

/// Appends every leaf board (depth == 0 and non-terminal) to `leaves` in the same
/// pre-order traversal that `minimax_node` will consume.
fn collect_leaves(board: &Board, depth: u32, leaves: &mut Vec<Board>) {
    let legal = generate_legal_moves(board);
    if legal.is_empty() {
        return; // terminal node — minimax_node handles it without a score slot
    }
    if depth == 0 {
        leaves.push(board.clone());
        return;
    }
    for &mv in &legal {
        collect_leaves(&board.make_move(mv), depth - 1, leaves);
    }
}

/// Walks the same tree as `collect_leaves` and propagates minimax values bottom-up.
/// Consumes exactly the leaf score slots that `collect_leaves` produced.
fn minimax_node(board: &Board, depth: u32, scores: &[f32], idx: &mut usize) -> f32 {
    let legal = generate_legal_moves(board);
    if legal.is_empty() {
        return terminal_value(board);
    }
    if depth == 0 {
        let v = scores[*idx];
        *idx += 1;
        return v;
    }
    match board.side_to_move {
        Color::White => legal
            .iter()
            .map(|&mv| minimax_node(&board.make_move(mv), depth - 1, scores, idx))
            .fold(f32::NEG_INFINITY, f32::max),
        Color::Black => legal
            .iter()
            .map(|&mv| minimax_node(&board.make_move(mv), depth - 1, scores, idx))
            .fold(f32::INFINITY, f32::min),
    }
}

pub fn parse_uci_move(board: &Board, s: &str) -> Option<Move> {
    if s.len() < 4 || !s.is_ascii() {
        return None;
    }
    let from = Square::from_algebraic(&s[0..2])?;
    let to = Square::from_algebraic(&s[2..4])?;
    let promo = s.chars().nth(4).and_then(|c| match c {
        'q' => Some(PieceKind::Queen),
        'r' => Some(PieceKind::Rook),
        'b' => Some(PieceKind::Bishop),
        'n' => Some(PieceKind::Knight),
        _ => None,
    });
    if s.len() > 4 && promo.is_none() {
        return None;
    }
    generate_legal_moves(board)
        .into_iter()
        .find(|mv| mv.from == from && mv.to == to && mv.promotion == promo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_e2e4_from_start() {
        let board = Board::starting_position();
        let mv = parse_uci_move(&board, "e2e4");
        assert!(mv.is_some());
        let mv = mv.unwrap();
        assert_eq!(mv.from, Square::from_algebraic("e2").unwrap());
        assert_eq!(mv.to, Square::from_algebraic("e4").unwrap());
        assert_eq!(mv.promotion, None);
    }

    #[test]
    fn parse_promotion_move() {
        let board = chess::fen::from_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1").unwrap();
        let mv = parse_uci_move(&board, "e7e8q");
        assert!(mv.is_some());
        let mv = mv.unwrap();
        assert_eq!(mv.from, Square::from_algebraic("e7").unwrap());
        assert_eq!(mv.to, Square::from_algebraic("e8").unwrap());
        assert_eq!(mv.promotion, Some(PieceKind::Queen));
    }

    #[test]
    fn parse_illegal_move_returns_none() {
        let board = Board::starting_position();
        assert!(parse_uci_move(&board, "e2e5").is_none());
    }

    #[test]
    fn parse_malformed_returns_none() {
        let board = Board::starting_position();
        assert!(parse_uci_move(&board, "xyz").is_none());
        assert!(parse_uci_move(&board, "").is_none());
    }

    #[test]
    fn parse_unknown_promo_char_returns_none() {
        let board = chess::fen::from_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1").unwrap();
        assert!(parse_uci_move(&board, "e7e8x").is_none());
    }

    #[test]
    fn best_move_returns_legal_move() {
        let model = HybridValueNet::new();
        model.set_training(false);
        let board = Board::starting_position();
        let legal = generate_legal_moves(&board);
        let result = best_move(&model, &board, 1);
        assert!(result.is_some());
        let (mv, score) = result.unwrap();
        assert!(legal.contains(&mv));
        assert!(score.is_finite());
    }

    #[test]
    fn best_move_returns_none_when_no_legal_moves() {
        let board =
            chess::fen::from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
                .unwrap();
        let model = HybridValueNet::new();
        model.set_training(false);
        assert!(best_move(&model, &board, 1).is_none());
    }

    // Verify alpha-beta finds a forced mate: White queen on h5 mates in 1 via f7.
    // Ignored by default — requires GPU and ~60 s cold-start.
    #[test]
    #[ignore = "requires GPU and ~60 s cold-start"]
    fn alpha_beta_finds_mate_in_one() {
        // Ruy Lopez-ish: Qh5 threatens Qxf7#
        let board = chess::fen::from_fen(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        )
        .unwrap();
        let model = HybridValueNet::new();
        model.set_training(false);
        let result = best_move(&model, &board, 2);
        assert!(result.is_some());
        let (mv, _) = result.unwrap();
        // Qxf7# is the only mating move
        assert_eq!(mv.from, Square::from_algebraic("h5").unwrap());
        assert_eq!(mv.to, Square::from_algebraic("f7").unwrap());
    }
}
