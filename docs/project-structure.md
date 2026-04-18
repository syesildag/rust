# Project Structure

This project is a **Cargo workspace** — a single repository containing multiple Rust
packages (called *crates*) that share code and are built together.

```
rust/
├── .cargo/
│   └── config.toml              # Custom cargo aliases (lint, check-all)
│
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions: fmt, clippy, tests on every push/PR
│
├── crates/
│   ├── core/                    # Shared utilities library
│   │   ├── src/
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   │
│   ├── chess/                   # Chess engine library (bitboard)
│   │   ├── src/
│   │   │   ├── lib.rs           # Public re-exports
│   │   │   ├── square.rs        # Square(u8) newtype
│   │   │   ├── piece.rs         # Color, PieceKind, Piece
│   │   │   ├── bitboard.rs      # u64 helpers
│   │   │   ├── attack.rs        # Precomputed attack tables + ray attacks
│   │   │   ├── moves.rs         # Move struct, MoveKind
│   │   │   ├── board.rs         # Board state, make_move, is_in_check
│   │   │   ├── fen.rs           # FEN parsing and serialisation
│   │   │   ├── movegen.rs       # Legal move generation, perft
│   │   │   └── game.rs          # GameStatus detection
│   │   └── Cargo.toml
│   │
│   └── cli/                     # Chess application binary
│       ├── src/
│       │   └── main.rs          # fn main — accepts FEN, prints moves + status
│       └── Cargo.toml
│
├── docs/                        # Project documentation
│   ├── chess-engine.md          # Chess engine technical reference
│   ├── crates.md                # Crate API overview
│   ├── getting-started.md       # First run and daily workflow
│   ├── project-structure.md     # This file
│   ├── ci-cd.md                 # CI/CD pipeline reference
│   └── tooling.md               # Linting, formatting, and toolchain notes
│
├── Cargo.toml                   # Workspace root: shared version, edition, lint rules
├── Cargo.lock                   # Locked dependency versions
├── rust-toolchain.toml          # Pins stable toolchain with rustfmt + clippy
└── rustfmt.toml                 # Code formatting configuration
```

---

## Key Concepts

### Workspace vs Crate

A **crate** is a single Rust package — it produces either a library (`.rlib`) or a binary.
A **workspace** is a collection of crates under one root `Cargo.toml` that share a single
`target/` build directory and `Cargo.lock` file.

### Crate responsibilities

| Crate | Type | Purpose |
|---|---|---|
| `core` | Library | General-purpose utilities, no domain logic |
| `chess` | Library | Complete chess engine — all game logic lives here |
| `cli` | Binary | Entry point — wires `chess` into a runnable program |

`chess` does not depend on `cli`. This separation means the engine could later power a
web server, TUI, or GUI without changing any engine code.

### The `target/` directory

Compiled output lives in `target/`. It is gitignored and can be regenerated at any time.
It can grow large (several GB) — `cargo clean` removes it entirely.
