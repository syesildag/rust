mod search;

use chess::board::Board;
use engine::model::HybridValueNet;
use engine::persist::Persist;
use search::{best_move, parse_uci_move};
use std::io::{self, BufRead, Write};
use tracing::warn;

const ENGINE_NAME: &str = "HybridNet";
const ENGINE_AUTHOR: &str = "serkan";

fn find_model_path() -> std::path::PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("model.bin");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    std::path::PathBuf::from("model.bin")
}

struct UciEngine {
    model: HybridValueNet,
    board: Board,
}

impl UciEngine {
    fn new() -> Self {
        let model_path = find_model_path();
        let model = HybridValueNet::load_from(&model_path).unwrap_or_else(|e| {
            warn!(error = %e, path = %model_path.display(), "no saved model — using random weights");
            HybridValueNet::default()
        });
        model.set_training(false);
        Self {
            model,
            board: Board::starting_position(),
        }
    }

    fn handle_position(&mut self, tokens: &[&str]) {
        let moves_idx = tokens.iter().position(|&t| t == "moves");
        let move_tokens = moves_idx.map_or(&[][..], |i| &tokens[i + 1..]);

        self.board = if tokens.first() == Some(&"startpos") {
            Board::starting_position()
        } else if tokens.first() == Some(&"fen") {
            let fen_end = moves_idx.unwrap_or(tokens.len());
            let fen = tokens[1..fen_end].join(" ");
            chess::fen::from_fen(&fen).unwrap_or_else(|e| {
                warn!(error = %e, %fen, "invalid FEN — using starting position");
                Board::starting_position()
            })
        } else {
            Board::starting_position()
        };

        for mv_str in move_tokens {
            if let Some(mv) = parse_uci_move(&self.board, mv_str) {
                self.board = self.board.make_move(mv);
            }
        }
    }
}

fn main() {
    tracing_subscriber::fmt().with_writer(io::stderr).init();

    let mut engine = UciEngine::new();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.as_slice() {
            ["uci", ..] => {
                writeln!(out, "id name {ENGINE_NAME}").ok();
                writeln!(out, "id author {ENGINE_AUTHOR}").ok();
                writeln!(out, "uciok").ok();
                out.flush().ok();
            }
            ["isready", ..] => {
                writeln!(out, "readyok").ok();
                out.flush().ok();
            }
            ["ucinewgame", ..] => {
                engine.board = Board::starting_position();
            }
            ["position", rest @ ..] => {
                engine.handle_position(rest);
            }
            ["go", ..] => {
                match best_move(&engine.model, &engine.board) {
                    Some((mv, eval)) => {
                        let mv_str = mv.to_string();
                        #[allow(clippy::cast_possible_truncation)]
                        let score_cp = (eval * 1000.0).round() as i32;
                        writeln!(out, "info depth 1 score cp {score_cp} pv {mv_str}").ok();
                        writeln!(out, "bestmove {mv_str}").ok();
                    }
                    None => {
                        writeln!(out, "bestmove 0000").ok();
                    }
                }
                out.flush().ok();
            }
            ["quit", ..] => break,
            _ => {}
        }
    }
}
