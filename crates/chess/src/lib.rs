//! Chess engine with pure bitboard representation.
//!
//! # Quick start
//!
//! ```
//! use chess::board::Board;
//! use chess::movegen::generate_legal_moves;
//! use chess::game::game_status;
//!
//! let board = Board::starting_position();
//! let moves = generate_legal_moves(&board);
//! assert_eq!(moves.len(), 20);
//! println!("Status: {:?}", game_status(&board));
//! ```

pub mod attack;
pub mod bitboard;
pub mod board;
pub mod fen;
pub mod game;
pub mod movegen;
pub mod moves;
pub mod piece;
pub mod square;
