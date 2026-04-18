# Getting Started

## Prerequisites

Install Rust via `rustup` — the official Rust toolchain manager:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Restart your terminal, then verify:

```bash
rustc --version   # the Rust compiler
cargo --version   # the Rust build tool & package manager
```

> **Note:** `rust-toolchain.toml` in the project root automatically tells `rustup` which
> toolchain version and components (rustfmt, clippy) to use when you run any `cargo` command
> inside this project. You don't need to install anything else manually.

---

## First Run

```bash
cargo run -p cli
```

This compiles the workspace and runs the `cli` binary. You should see:

```
6 + 7 = 13
```

---

## Running Tests

```bash
cargo test --all
```

This runs:
- Unit tests inside each crate (the `#[test]` functions)
- Doc tests — the code examples in `///` comments are compiled and executed as tests

Expected output:

```
running 1 test
test tests::test_add ... ok

test result: ok. 1 passed; 0 failed
```

---

## Daily Workflow

| Task | Command |
|---|---|
| Compile and run | `cargo run -p cli` |
| Run all tests | `cargo test --all` |
| Auto-format code | `cargo fmt` |
| Check formatting (without changing files) | `cargo fmt --check` |
| Run the linter | `cargo lint` |
| Run fmt + lint together | `cargo check-all` |
| Build without running | `cargo build --all` |
| Generate and open API docs | `cargo doc --open` |

`cargo lint` and `cargo check-all` are custom aliases defined in [`.cargo/config.toml`](../.cargo/config.toml).
