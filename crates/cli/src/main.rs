use chess::board::Board;
use chess::fen::from_fen;
use chess::game::{game_status, GameStatus};
use chess::movegen::generate_legal_moves;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let board = if args.len() == 1 {
        Board::starting_position()
    } else if args.len() == 7 {
        // FEN has 6 space-separated fields passed as separate shell args
        let fen = args[1..].join(" ");
        match from_fen(&fen) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Invalid FEN: {e}");
                std::process::exit(1);
            }
        }
    } else {
        eprintln!(
            "Usage: {} [piece_placement side castling ep halfmove fullmove]",
            args[0]
        );
        eprintln!(
            "Example: {} rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            args[0]
        );
        std::process::exit(1);
    };

    let moves = generate_legal_moves(&board);
    let status = game_status(&board);

    println!("FEN: {}", board.to_fen());
    println!("Side to move: {:?}", board.side_to_move);
    println!(
        "Legal moves ({}): {}",
        moves.len(),
        moves
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    match status {
        GameStatus::Ongoing => println!("Status: Ongoing"),
        GameStatus::Checkmate => println!("Status: Checkmate"),
        GameStatus::Stalemate => println!("Status: Stalemate"),
        GameStatus::Draw(reason) => println!("Status: Draw ({reason:?})"),
    }
}
