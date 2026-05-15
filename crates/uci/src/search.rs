use chess::board::Board;
use chess::moves::Move;
use engine::model::HybridValueNet;

pub fn best_move(_model: &HybridValueNet, _board: &Board) -> Option<Move> {
    todo!()
}

pub fn parse_uci_move(_board: &Board, _s: &str) -> Option<Move> {
    todo!()
}
