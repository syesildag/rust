use chess::board::Board;
use chess::movegen::generate_legal_moves;
use chess::moves::Move;
use chess::piece::{Color, PieceKind};
use chess::square::Square;
use engine::model::HybridValueNet;
use tracing::info;

pub fn best_move(model: &HybridValueNet, board: &Board) -> Option<(Move, f32)> {
    let legal = generate_legal_moves(board);
    if legal.is_empty() {
        return None;
    }
    let after_boards: Vec<Board> = legal
        .iter()
        .copied()
        .map(|mv| board.make_move(mv))
        .collect();
    info!(n = after_boards.len(), "forward_batch: start");
    let raw = model.forward_batch(&after_boards).data();
    info!("forward_batch: done");
    let sign = match board.side_to_move {
        Color::White => 1.0_f32,
        Color::Black => -1.0_f32,
    };
    (0..legal.len())
        .max_by(|&i, &j| {
            (sign * raw[i])
                .partial_cmp(&(sign * raw[j]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|i| (legal[i], sign * raw[i]))
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
        assert!(parse_uci_move(&board, "e2e5").is_none()); // illegal jump
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
        let result = best_move(&model, &board);
        assert!(result.is_some());
        let (mv, score) = result.unwrap();
        assert!(legal.contains(&mv));
        assert!(score.is_finite());
    }

    #[test]
    fn best_move_returns_none_when_no_legal_moves() {
        // Checkmate position: Fool's Mate — White is checkmated.
        let board =
            chess::fen::from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
                .unwrap();
        let model = HybridValueNet::new();
        model.set_training(false);
        assert!(best_move(&model, &board).is_none());
    }
}
