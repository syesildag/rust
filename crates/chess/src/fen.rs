use std::fmt;
use std::fmt::Write as _;
use std::num::ParseIntError;

use crate::board::Board;
use crate::piece::{Color, Piece, PieceKind};
use crate::square::Square;

/// Errors that can occur while parsing a FEN string.
#[derive(Debug)]
pub enum FenError {
    WrongFieldCount(usize),
    InvalidPieceChar(char),
    InvalidSquare(String),
    InvalidCastling(String),
    InvalidSideToMove(char),
    ParseInt(ParseIntError),
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongFieldCount(n) => write!(f, "expected 6 FEN fields, got {n}"),
            Self::InvalidPieceChar(c) => write!(f, "invalid piece character '{c}'"),
            Self::InvalidSquare(s) => write!(f, "invalid square '{s}'"),
            Self::InvalidCastling(s) => write!(f, "invalid castling rights '{s}'"),
            Self::InvalidSideToMove(c) => write!(f, "invalid side to move '{c}'"),
            Self::ParseInt(e) => write!(f, "integer parse error: {e}"),
        }
    }
}

impl From<ParseIntError> for FenError {
    fn from(e: ParseIntError) -> Self {
        Self::ParseInt(e)
    }
}

/// Parses a FEN string into a `Board`.
///
/// # Errors
/// Returns a `FenError` if any field of the FEN string is invalid.
pub fn from_fen(s: &str) -> Result<Board, FenError> {
    let fields: Vec<&str> = s.split_whitespace().collect();
    if fields.len() != 6 {
        return Err(FenError::WrongFieldCount(fields.len()));
    }

    let mut pieces = [[0u64; 6]; 2];

    // Field 1: piece placement
    let mut rank: u8 = 7;
    let mut file: u8 = 0;
    for ch in fields[0].chars() {
        match ch {
            '/' => {
                if rank == 0 {
                    return Err(FenError::InvalidSquare(fields[0].to_string()));
                }
                rank -= 1;
                file = 0;
            }
            '1'..='8' => {
                file += ch as u8 - b'0';
            }
            _ => {
                let (color, kind) = piece_from_char(ch)?;
                let sq = Square::from_file_rank(file, rank);
                pieces[color as usize][kind.index()] |= sq.bit();
                file += 1;
            }
        }
    }

    // Field 2: side to move
    let side_to_move = match fields[1] {
        "w" => Color::White,
        "b" => Color::Black,
        s => return Err(FenError::InvalidSideToMove(s.chars().next().unwrap_or('?'))),
    };

    // Field 3: castling rights (bits: 0=WK, 1=WQ, 2=BK, 3=BQ)
    let castling = if fields[2] == "-" {
        0u8
    } else {
        let mut c = 0u8;
        for ch in fields[2].chars() {
            match ch {
                'K' => c |= 1,
                'Q' => c |= 2,
                'k' => c |= 4,
                'q' => c |= 8,
                _ => return Err(FenError::InvalidCastling(fields[2].to_string())),
            }
        }
        c
    };

    // Field 4: en passant square
    let en_passant = if fields[3] == "-" {
        None
    } else {
        Some(
            Square::from_algebraic(fields[3])
                .ok_or_else(|| FenError::InvalidSquare(fields[3].to_string()))?,
        )
    };

    // Fields 5 & 6: halfmove clock, fullmove number
    let halfmove_clock: u8 = fields[4].parse()?;
    let fullmove_number: u16 = fields[5].parse()?;

    Ok(Board {
        pieces,
        side_to_move,
        castling,
        en_passant,
        halfmove_clock,
        fullmove_number,
    })
}

fn piece_from_char(ch: char) -> Result<(Color, PieceKind), FenError> {
    let color = if ch.is_uppercase() {
        Color::White
    } else {
        Color::Black
    };
    let kind = match ch.to_ascii_lowercase() {
        'p' => PieceKind::Pawn,
        'n' => PieceKind::Knight,
        'b' => PieceKind::Bishop,
        'r' => PieceKind::Rook,
        'q' => PieceKind::Queen,
        'k' => PieceKind::King,
        _ => return Err(FenError::InvalidPieceChar(ch)),
    };
    Ok((color, kind))
}

/// Converts a `Board` back to a FEN string.
#[must_use]
pub fn to_fen(board: &Board) -> String {
    let mut s = String::new();

    // Piece placement
    for rank in (0..8).rev() {
        let mut empty = 0u8;
        for file in 0..8u8 {
            let sq = Square::from_file_rank(file, rank);
            let piece = board.piece_at(sq);
            if let Some(p) = piece {
                if empty > 0 {
                    s.push((b'0' + empty) as char);
                    empty = 0;
                }
                let ch = p.kind.fen_char();
                s.push(if p.color == Color::White {
                    ch
                } else {
                    ch.to_ascii_lowercase()
                });
            } else {
                empty += 1;
            }
        }
        if empty > 0 {
            s.push((b'0' + empty) as char);
        }
        if rank > 0 {
            s.push('/');
        }
    }

    s.push(' ');
    s.push(if board.side_to_move == Color::White {
        'w'
    } else {
        'b'
    });
    s.push(' ');

    // Castling rights
    if board.castling == 0 {
        s.push('-');
    } else {
        if board.castling & 1 != 0 {
            s.push('K');
        }
        if board.castling & 2 != 0 {
            s.push('Q');
        }
        if board.castling & 4 != 0 {
            s.push('k');
        }
        if board.castling & 8 != 0 {
            s.push('q');
        }
    }

    s.push(' ');
    match board.en_passant {
        Some(sq) => s.push_str(&sq.to_string()),
        None => s.push('-'),
    }

    let _ = write!(s, " {} {}", board.halfmove_clock, board.fullmove_number);
    s
}

/// Parses a `Piece` from a FEN character (for public use).
///
/// # Errors
/// Returns `FenError::InvalidPieceChar` if `ch` is not a valid FEN piece character.
pub fn parse_piece(ch: char) -> Result<Piece, FenError> {
    let (color, kind) = piece_from_char(ch)?;
    Ok(Piece::new(kind, color))
}

#[cfg(test)]
mod tests {
    use super::*;

    const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn parse_starting_fen() {
        let board = from_fen(START_FEN).unwrap();
        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(board.castling, 0b0000_1111);
        assert!(board.en_passant.is_none());
        assert_eq!(board.halfmove_clock, 0);
        assert_eq!(board.fullmove_number, 1);
    }

    #[test]
    fn fen_round_trip() {
        let board = from_fen(START_FEN).unwrap();
        assert_eq!(to_fen(&board), START_FEN);
    }

    #[test]
    fn invalid_piece_char() {
        assert!(matches!(
            from_fen("xnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            Err(FenError::InvalidPieceChar('x'))
        ));
    }

    #[test]
    fn wrong_field_count() {
        assert!(matches!(
            from_fen("rnbqkbnr w"),
            Err(FenError::WrongFieldCount(2))
        ));
    }

    #[test]
    fn en_passant_fen() {
        let fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3";
        let board = from_fen(fen).unwrap();
        assert_eq!(board.en_passant, Square::from_algebraic("f6"));
    }
}
