//! FEN-file parser: one labelled position per line.
//!
//! ## Supported line formats
//!
//! ```text
//! # comment — ignored
//! <FEN>                          ← position only, outcome defaults to 0.0 (draw)
//! <FEN> 1-0                      ← PGN-style result token
//! <FEN> 0-1
//! <FEN> 1/2-1/2
//! <FEN> 1.0                      ← float outcome
//! <FEN> 0.0
//! <FEN> -1.0
//! ```
//!
//! A FEN string has exactly 6 space-separated fields, so the optional outcome
//! is always the 7th token on the line.
//!
//! Lines that fail FEN parsing emit a `tracing::warn!` and are skipped.

use crate::pgn::Sample;
use chess::fen::from_fen;

/// Parses a FEN file and returns `(Board, outcome)` pairs.
#[must_use]
pub fn parse_fen_file(text: &str) -> Vec<Sample> {
    text.lines().filter_map(parse_line).collect()
}

/// Parses a CSV file where each line is `<FEN>, <outcome>`.
///
/// The FEN is the standard 6-field string; the outcome is separated from it by
/// the first comma on the line.  `game_id` is always `0` because CSV files
/// carry no game context.  Blank lines and `#`-comments are skipped; lines
/// with an invalid FEN or an out-of-range outcome are also skipped with a
/// `tracing::warn!`.
#[must_use]
pub fn parse_csv_file(text: &str) -> Vec<Sample> {
    text.lines().filter_map(parse_csv_line).collect()
}

fn parse_csv_line(line: &str) -> Option<Sample> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    let (fen_part, outcome_part) = line.split_once(',')?;
    let fen = fen_part.trim();
    let board = match from_fen(fen) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(fen = fen, error = %e, "invalid FEN in CSV line skipped");
            return None;
        }
    };

    let outcome = parse_outcome(outcome_part.trim())?;
    Some((board, outcome, 0u64))
}

fn parse_line(line: &str) -> Option<Sample> {
    let line = line.trim();
    // Skip comments and empty lines.
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    // A FEN has exactly 6 fields; the outcome is an optional 7th token.
    let tokens: Vec<&str> = line.splitn(8, ' ').collect();
    if tokens.len() < 6 {
        return None;
    }

    let fen: String = tokens[..6].join(" ");
    let board = match from_fen(&fen) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(fen = fen.as_str(), error = %e, "invalid FEN line skipped");
            return None;
        }
    };

    let outcome = if tokens.len() >= 7 {
        parse_outcome(tokens[6])
    } else {
        Some(0.0f32) // no label → treat as draw
    };

    Some((board, outcome?, 0u64))
}

fn parse_outcome(s: &str) -> Option<f32> {
    match s {
        "1-0" => Some(1.0),
        "0-1" => Some(-1.0),
        "1/2-1/2" => Some(0.0),
        "*" => None, // unknown result — skip
        other => other
            .parse::<f32>()
            .ok()
            .filter(|&v| (-1.0..=1.0).contains(&v)),
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn csv_parses_float_outcome() {
        let text = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1, 0.24\n";
        let samples = parse_csv_file(text);
        assert_eq!(samples.len(), 1);
        assert!((samples[0].1 - 0.24).abs() < 1e-5);
        assert_eq!(samples[0].2, 0u64); // game_id is always 0
    }

    #[test]
    fn csv_parses_multiple_lines() {
        let text = "\
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1, 0.24
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1, 0.59
";
        let samples = parse_csv_file(text);
        assert_eq!(samples.len(), 2);
        assert!((samples[1].1 - 0.59).abs() < 1e-5);
    }

    #[test]
    fn csv_skips_comments_and_blanks() {
        let text = "\
# comment

rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1, 0.5
";
        let samples = parse_csv_file(text);
        assert_eq!(samples.len(), 1);
    }

    #[test]
    fn csv_skips_invalid_fen() {
        let text = "not-a-fen, 0.5\n";
        let samples = parse_csv_file(text);
        assert!(samples.is_empty());
    }

    #[test]
    fn csv_skips_out_of_range_outcome() {
        let text = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1, 99.0\n";
        let samples = parse_csv_file(text);
        assert!(samples.is_empty());
    }

    #[test]
    fn csv_skips_missing_comma() {
        let text = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n";
        let samples = parse_csv_file(text);
        assert!(samples.is_empty());
    }

    #[test]
    fn parses_pgn_style_outcomes() {
        let text = "\
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 1-0
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 0-1
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 1/2-1/2
";
        let samples = parse_fen_file(text);
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].1, 1.0);
        assert_eq!(samples[1].1, -1.0);
        assert_eq!(samples[2].1, 0.0);
    }

    #[test]
    fn parses_float_outcomes() {
        let text = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 0.75\n";
        let samples = parse_fen_file(text);
        assert_eq!(samples.len(), 1);
        assert!((samples[0].1 - 0.75).abs() < 1e-6);
    }

    #[test]
    fn skips_comments_and_blanks() {
        let text = "\
# this is a comment

rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 0.0
";
        let samples = parse_fen_file(text);
        assert_eq!(samples.len(), 1);
    }

    #[test]
    fn unknown_result_skipped() {
        let text = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 *\n";
        let samples = parse_fen_file(text);
        assert!(samples.is_empty());
    }

    #[test]
    fn no_outcome_defaults_to_draw() {
        let text = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n";
        let samples = parse_fen_file(text);
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].1, 0.0);
    }
}
