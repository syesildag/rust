# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
cargo run -p cli              # compile and run the CLI binary
cargo build --all             # compile all crates without running
cargo run -p cli -- 3 5       # pass arguments to the CLI (prints "3 + 5 = 8")
```

## Testing

```bash
cargo test --all              # run all unit tests and doc tests across all crates
cargo test -p core            # run tests for a single crate
cargo test test_add           # run a single test by name
```

Doc-comment code blocks (`///` with a code block) are compiled and executed as doc tests by `cargo test`.

## Linting & Formatting

```bash
cargo fmt                     # auto-format all files in place
cargo fmt --check             # check formatting without modifying (used in CI)
cargo lint                    # alias: cargo clippy --all-targets --all-features -- -D warnings
cargo check-all               # alias: cargo fmt --check && cargo lint (run before pushing)
```

`cargo lint` and `cargo check-all` are defined in `.cargo/config.toml` and are local to this project.

## Architecture

This is a **Cargo workspace** with two crates under `crates/`:

- **`crates/core`** — library crate (`lib.rs`). Contains pure logic with no `main`. Designed to be reusable by multiple consumers.
- **`crates/cli`** — binary crate (`main.rs`). Depends on `core` via a local path dependency. Responsible only for parsing arguments and presenting output.

The workspace root `Cargo.toml` defines shared `version`, `edition`, and lint rules. Each crate inherits them via `[lints] workspace = true`. Adding a new crate means creating its directory and adding it to the `members` list in the root `Cargo.toml`.

## Lint Rules

Workspace-wide lint configuration (root `Cargo.toml`):
- `unsafe_code = "forbid"` — `unsafe` blocks will not compile
- `clippy::pedantic = "warn"` — strict Clippy lints are enabled across all crates

Clippy warnings are promoted to errors in CI (`-D warnings`), so `cargo lint` must pass cleanly before pushing.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs three parallel jobs on every push to `main` and every PR: `fmt` (`cargo fmt --check`), `clippy` (`cargo lint`), and `test` (`cargo test --all`). The `rust-toolchain.toml` pins the toolchain to `stable` with `rustfmt` and `clippy` components.
