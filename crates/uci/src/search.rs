use chess::board::Board;
use chess::movegen::generate_legal_moves;
use chess::moves::Move;
use chess::piece::PieceKind;
use chess::square::Square;
use engine::model::HybridValueNet;

#[allow(dead_code)]
#[must_use]
pub fn best_move(_model: &HybridValueNet, _board: &Board) -> Option<Move> {
    todo!()
}

#[allow(dead_code)]
#[must_use]
pub fn parse_uci_move(board: &Board, s: &str) -> Option<Move> {
    if s.len() < 4 {
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
        // White pawn on e7, ready to promote.
        let board = chess::fen::from_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1").unwrap();
        let mv = parse_uci_move(&board, "e7e8q");
        assert!(mv.is_some());
        let mv = mv.unwrap();
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
}
