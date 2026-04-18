# Getting Started

## Prerequisites

Install Rust via `rustup` — the official Rust toolchain manager:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Restart your terminal, then verify:

```bash
rustc --version
cargo --version
```

> `rust-toolchain.toml` automatically tells `rustup` which toolchain version and components
> (rustfmt, clippy) to use inside this project. Nothing else needs to be installed.

---

## First Run

Run the chess CLI with the starting position:

```bash
cargo run -p cli
```

Expected output:

```
FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Side to move: White
Legal moves (20): a2a3, a2a4, b2b3, b2b4, c2c3, c2c4, d2d3, d2d4, e2e3, e2e4, f2f3, f2f4, g2g3, g2g4, h2h3, h2h4, b1a3, b1c3, g1f3, g1h3
Status: Ongoing
```

### Custom position via FEN

Pass the six FEN fields as separate shell arguments:

```bash
cargo run -p cli -- rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
```

### Checkmate example

```bash
# Scholar's mate — white is in checkmate
cargo run -p cli -- "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
# Status: Checkmate
```

---

## Running Tests

```bash
cargo test --all
```

This runs all unit tests, doc tests, and move-generation correctness tests (perft) across
all crates. The perft tests verify the engine generates exactly the right number of legal
moves at depth 1, 2, and 3 from the starting position (20 / 400 / 8902).

---

## Daily Workflow

| Task | Command |
|---|---|
| Compile and run | `cargo run -p cli` |
| Run all tests | `cargo test --all` |
| Run chess tests only | `cargo test -p chess` |
| Auto-format code | `cargo fmt` |
| Check formatting | `cargo fmt --check` |
| Run the linter | `cargo lint` |
| Run fmt + lint together | `cargo check-all` |
| Build without running | `cargo build --all` |
| Generate and open API docs | `cargo doc -p chess --open` |

`cargo lint` and `cargo check-all` are custom aliases defined in [`.cargo/config.toml`](../.cargo/config.toml).
