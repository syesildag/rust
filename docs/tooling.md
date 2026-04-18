# Tooling

This project uses several tools that work together to keep code correct, consistent, and
maintainable. All of them ship with `rustup` — no separate installs needed.

---

## cargo

`cargo` is Rust's all-in-one build tool. Unlike most languages (where build, test, format,
and lint are separate tools), cargo does everything:

```bash
cargo build     # compile
cargo run       # compile + run
cargo test      # compile + run tests
cargo fmt       # format code
cargo clippy    # lint
cargo doc       # generate documentation
cargo add       # add a dependency to Cargo.toml
```

---

## rustfmt — Code Formatter

`rustfmt` automatically formats Rust code to a consistent style. There's no debate about
tabs vs spaces or brace placement — the formatter decides.

Configuration: [`rustfmt.toml`](../rustfmt.toml)

```toml
edition = "2021"
```

**Usage:**

```bash
cargo fmt            # format all files in place
cargo fmt --check    # check without modifying (used in CI)
```

Run `cargo fmt` before every commit so formatting is never a review issue.

---

## Clippy — Linter

Clippy catches common mistakes, unnecessary code, and unidiomatic patterns that the
compiler itself won't reject. It's like having an experienced Rust developer review
your code automatically.

Example: Clippy will warn if you write `x == true` instead of just `x`, or if you use a
`Vec` where a slice would be more idiomatic.

**Usage:**

```bash
cargo lint    # alias for: cargo clippy --all-targets --all-features -- -D warnings
```

The `-D warnings` flag makes Clippy warnings into errors — the build fails if any lint
fires. This is intentional: it keeps the codebase clean from the start.

---

## Cargo Aliases

Defined in [`.cargo/config.toml`](../.cargo/config.toml), these are project-local shortcuts:

| Alias | Expands to | Purpose |
|---|---|---|
| `cargo lint` | `cargo clippy --all-targets --all-features -- -D warnings` | Run Clippy with strict settings |
| `cargo check-all` | `cargo fmt --check && cargo lint` | Pre-commit check: fmt + lint |

Run `cargo check-all` before pushing to catch issues before CI does.

---

## rust-toolchain.toml

Pins the exact Rust toolchain for this project:

```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
```

When you run any `cargo` command inside this project, `rustup` automatically uses the
pinned toolchain. This means every developer (and CI) uses the same compiler version,
eliminating "it works on my machine" problems.

---

## Workspace Lints

Lint rules are configured once in the root [`Cargo.toml`](../Cargo.toml) and inherited by
all crates:

```toml
[workspace.lints.rust]
unsafe_code = "forbid"    # using `unsafe` blocks will not compile

[workspace.lints.clippy]
pedantic = "warn"         # enables a stricter set of Clippy lints
```

Each crate opts in via `[lints] workspace = true` in its own `Cargo.toml`.
