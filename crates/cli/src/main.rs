use chess::board::Board;
use chess::fen::from_fen;
use chess::game::{game_status, GameStatus};
use chess::movegen::generate_legal_moves;
use std::path::PathBuf;
use tracing::info_span;
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(true)
        .with_span_events(FmtSpan::CLOSE)
        .init();

    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(String::as_str) {
        Some("train") => {
            let _span = info_span!("train").entered();
            cmd_train(&args[2..]);
        }
        Some("selfplay") => {
            let _span = info_span!("selfplay").entered();
            cmd_selfplay(&args[2..]);
        }
        Some("eval") => {
            let _span = info_span!("position-eval").entered();
            cmd_eval(&args[2..]);
        }
        _ => cmd_board(&args),
    }
}

// ─── train ───────────────────────────────────────────────────────────────────

fn cmd_train(args: &[String]) {
    use engine::train::{train, TrainConfig};

    let mut cfg = TrainConfig::default();
    cfg.pgn_paths.clear(); // we'll fill from --games flags

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            // --games accepts one or more paths; keep consuming until the next flag
            "--games" => {
                i += 1;
                while i < args.len() && !args[i].starts_with("--") {
                    cfg.pgn_paths.push(PathBuf::from(&args[i]));
                    i += 1;
                }
                continue; // already advanced i
            }
            "--epochs" => {
                i += 1;
                cfg.epochs = args[i].parse().unwrap_or(20);
            }
            "--batch" => {
                i += 1;
                cfg.batch_size = args[i].parse().unwrap_or(32);
            }
            "--lr" => {
                i += 1;
                cfg.lr = args[i].parse().unwrap_or(1e-4);
            }
            "--output" => {
                i += 1;
                cfg.output = PathBuf::from(&args[i]);
            }
            _ => {}
        }
        i += 1;
    }

    // Fall back to default if no --games was given
    if cfg.pgn_paths.is_empty() {
        cfg.pgn_paths.push(PathBuf::from("games.pgn"));
    }

    println!("Loading games from {} source(s)…", cfg.pgn_paths.len());
    match train(cfg) {
        Ok(_model) => println!("Training complete."),
        Err(e) => {
            eprintln!("Train error: {e}");
            std::process::exit(1);
        }
    }
}

// ─── eval ────────────────────────────────────────────────────────────────────

fn cmd_eval(args: &[String]) {
    use engine::model::HybridValueNet;

    let fen_str = flag_value(args, "--fen");
    let board = if let Some(f) = fen_str {
        match from_fen(f) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Invalid FEN: {e}");
                std::process::exit(1);
            }
        }
    } else {
        Board::starting_position()
    };

    let model = HybridValueNet::new();
    let value = model.forward(&board).data()[0];

    println!("{board}");
    println!();
    println!("Evaluation: {value:+.4}  (positive = White advantage)");
}

// ─── selfplay ────────────────────────────────────────────────────────────────

fn cmd_selfplay(args: &[String]) {
    use engine::model::HybridValueNet;
    use engine::selfplay::generate;

    let n: usize = flag_value(args, "--games")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let model = HybridValueNet::new();
    let dataset = generate(&model, n);
    println!(
        "Self-play complete: {n} games → {} positions",
        dataset.len()
    );
}

// ─── board display (default) ─────────────────────────────────────────────────

fn cmd_board(args: &[String]) {
    let board = if args.len() == 1 {
        Board::starting_position()
    } else if args.len() == 7 {
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
            "Usage:\n  {} train    [--games FILE…] [--epochs N] [--lr F] [--output FILE]\n  \
             {} eval     [--fen \"FEN\"]\n  \
             {} selfplay [--games N]\n  \
             {} [FEN fields…]",
            args[0], args[0], args[0], args[0]
        );
        std::process::exit(1);
    };

    let moves = generate_legal_moves(&board);
    let status = game_status(&board);

    println!("{board}");
    println!();
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
        GameStatus::Draw(r) => println!("Status: Draw ({r:?})"),
    }
}

// ─── helper ──────────────────────────────────────────────────────────────────

fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}
