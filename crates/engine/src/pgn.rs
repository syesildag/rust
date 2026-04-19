//! Minimal PGN parser: extracts game sequences as `(Board, outcome)` pairs.
//!
//! Supports Standard Algebraic Notation (SAN) move disambiguation.  Moves that
//! cannot be parsed or replayed are silently skipped — this makes the parser
//! robust to annotations, NAG symbols (`$1`, `!?`, …), and clock tags `{ ... }`.
//!
//! ## Outcome encoding
//! | PGN result | label |
//! |------------|-------|
//! | `1-0`      |  1.0  |
//! | `0-1`      | -1.0  |
//! | `1/2-1/2`  |  0.0  |

use chess::board::Board;
use chess::movegen::generate_legal_moves;
use chess::moves::{Move, MoveKind};
use chess::piece::PieceKind;
use chess::square::Square;

/// One (position, outcome) sample extracted from a PGN game.
pub type Sample = (Board, f32);

// ─── public API ──────────────────────────────────────────────────────────────

/// Parses a PGN string and returns all (board, outcome) samples from every
/// game contained in the text.
#[must_use]
pub fn parse_pgn(pgn: &str) -> Vec<Sample> {
    games(pgn).into_iter().flat_map(parse_game).collect()
}

// ─── game splitting ───────────────────────────────────────────────────────────

/// Splits a multi-game PGN string into individual game strings.
fn games(pgn: &str) -> Vec<&str> {
    split_games(pgn)
}

/// Robust game splitter: each game starts at an `[Event` tag.
fn split_games(pgn: &str) -> Vec<&str> {
    let mut starts: Vec<usize> = pgn.match_indices("[Event ").map(|(i, _)| i).collect();
    starts.push(pgn.len());
    starts
        .windows(2)
        .map(|w| pgn[w[0]..w[1]].trim())
        .filter(|s| !s.is_empty())
        .collect()
}

// ─── single-game parser ───────────────────────────────────────────────────────

fn parse_game(pgn: &str) -> Vec<Sample> {
    let outcome = extract_outcome(pgn);
    let Some(outcome) = outcome else {
        return Vec::new();
    };

    let move_text = strip_headers(pgn);
    let tokens = tokenise_moves(&move_text);

    let mut board = Board::starting_position();
    let mut samples = Vec::new();

    for token in &tokens {
        if is_result_token(token) {
            break;
        }
        if let Some(mv) = san_to_move(&board, token) {
            samples.push((board.clone(), outcome));
            board = board.make_move(mv);
        }
    }
    samples
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn extract_outcome(pgn: &str) -> Option<f32> {
    if pgn.contains("1-0") {
        return Some(1.0);
    }
    if pgn.contains("0-1") {
        return Some(-1.0);
    }
    if pgn.contains("1/2-1/2") {
        return Some(0.0);
    }
    None
}

fn strip_headers(pgn: &str) -> String {
    pgn.lines()
        .filter(|l| !l.trim_start().starts_with('['))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Removes comments `{ ... }`, move numbers `1.`, and extra whitespace.
fn tokenise_moves(text: &str) -> Vec<String> {
    // Strip brace comments
    let mut clean = String::with_capacity(text.len());
    let mut depth = 0usize;
    for ch in text.chars() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
            }
            c if depth == 0 => clean.push(c),
            _ => {}
        }
    }
    // Split on whitespace, remove move numbers and NAGs
    clean
        .split_whitespace()
        .filter(|t| {
            !t.ends_with('.')          // "1.", "2.", "10."
            && !t.starts_with('$')     // NAG symbols
            && !t.is_empty()
        })
        .map(|t| t.trim_end_matches(['+', '#']).to_string())
        .collect()
}

fn is_result_token(t: &str) -> bool {
    matches!(t, "1-0" | "0-1" | "1/2-1/2" | "*")
}

// ─── SAN → Move ──────────────────────────────────────────────────────────────

/// Converts a SAN token to a legal `Move` on `board`, or `None` if unparseable.
fn san_to_move(board: &Board, san: &str) -> Option<Move> {
    let legal = generate_legal_moves(board);

    // Castling
    if san == "O-O-O" || san == "0-0-0" {
        return legal
            .iter()
            .copied()
            .find(|m| m.kind == MoveKind::Castling && m.to.file() < m.from.file());
    }
    if san == "O-O" || san == "0-0" {
        return legal
            .iter()
            .copied()
            .find(|m| m.kind == MoveKind::Castling && m.to.file() > m.from.file());
    }

    let bytes = san.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    // Determine piece kind from first character
    let (piece_kind, rest) = if bytes[0].is_ascii_uppercase() {
        let kind = match bytes[0] {
            b'N' => PieceKind::Knight,
            b'B' => PieceKind::Bishop,
            b'R' => PieceKind::Rook,
            b'Q' => PieceKind::Queen,
            b'K' => PieceKind::King,
            _ => return None,
        };
        (kind, &san[1..])
    } else {
        (PieceKind::Pawn, san)
    };

    // Strip promotion suffix (e.g. "=Q") to get destination
    let (rest_no_promo, promo_kind) = if let Some(eq) = rest.rfind('=') {
        let pk = match rest.as_bytes().get(eq + 1) {
            Some(b'R') => PieceKind::Rook,
            Some(b'B') => PieceKind::Bishop,
            Some(b'N') => PieceKind::Knight,
            _ => PieceKind::Queen,
        };
        (&rest[..eq], Some(pk))
    } else {
        (rest, None)
    };

    // Remove capture 'x'
    let rest_no_cap: String = rest_no_promo.chars().filter(|&c| c != 'x').collect();
    let r = rest_no_cap.as_str();

    // Last two chars should be the destination square (e.g. "e4", "d7")
    if r.len() < 2 {
        return None;
    }
    let dest_str = &r[r.len() - 2..];
    let dest = Square::from_algebraic(dest_str)?;

    // Optional disambiguator: the chars between piece letter (stripped) and dest
    let disambig = &r[..r.len() - 2];

    legal.iter().copied().find(|&mv| {
        if mv.to != dest {
            return false;
        }
        let Some(p) = board.piece_at(mv.from) else {
            return false;
        };
        if p.kind != piece_kind {
            return false;
        }
        if p.color != board.side_to_move {
            return false;
        }
        // Check promotion
        if mv.promotion != promo_kind && promo_kind.is_some() {
            return false;
        }
        // Check disambiguator
        if !disambig.is_empty() {
            let db = disambig.as_bytes();
            if db.len() == 1 {
                let d = db[0];
                if d.is_ascii_digit() {
                    let rank = d - b'1';
                    if mv.from.rank() != rank {
                        return false;
                    }
                } else {
                    let file = d - b'a';
                    if mv.from.file() != file {
                        return false;
                    }
                }
            } else if db.len() == 2 {
                let Some(sq) = Square::from_algebraic(disambig) else {
                    return false;
                };
                if mv.from != sq {
                    return false;
                }
            }
        }
        true
    })
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const MINI_PGN: &str = r#"[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"#;

    #[test]
    fn parse_scholars_mate() {
        let samples = parse_pgn(MINI_PGN);
        // 7 half-moves (positions before each move)
        assert_eq!(samples.len(), 7);
        // All samples have outcome 1.0
        assert!(samples.iter().all(|(_, v)| (*v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn castling_kingside() {
        // Ruy Lopez main line up to the point where White can castle
        let pgn = r#"[Event "X"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Nxe4 1-0
"#;
        let samples = parse_pgn(pgn);
        assert!(!samples.is_empty());
    }
}
