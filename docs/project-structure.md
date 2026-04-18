# Project Structure

This project is a **Cargo workspace** — a single repository that contains multiple Rust
packages (called *crates*) that can share code and be built together.

```
rust/
├── .cargo/
│   └── config.toml          # Custom cargo command aliases (lint, check-all)
│
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions: runs fmt, clippy, and tests on every push/PR
│
├── crates/                  # All crates live here
│   ├── core/                # Shared library — reusable logic
│   │   ├── src/
│   │   │   └── lib.rs       # Library entry point
│   │   └── Cargo.toml       # core crate manifest
│   │
│   └── cli/                 # Binary — the runnable program
│       ├── src/
│       │   └── main.rs      # Program entry point (fn main)
│       └── Cargo.toml       # cli crate manifest
│
├── Cargo.toml               # Workspace root: shared version, edition, lint rules
├── Cargo.lock               # Locked dependency versions (committed for binaries)
├── rust-toolchain.toml      # Pins the Rust toolchain version for this project
├── rustfmt.toml             # Code formatting configuration
└── .gitignore               # Ignores the build output directory (target/)
```

---

## Key Concepts

### Workspace vs Crate

A **crate** is a single Rust package — it produces either a library (`.rlib`) or a binary
(executable). A **workspace** is a collection of crates under one root `Cargo.toml` that
share a single `target/` build directory and `Cargo.lock` file.

Think of the workspace as a monorepo and each crate as a project within it.

### Why split into `core` and `cli`?

| Crate | Type | Purpose |
|---|---|---|
| `core` | Library | Contains pure logic — functions, data types. Has no `main`. |
| `cli` | Binary | The entry point. Wires together `core` functions into a runnable program. |

This separation means `core` could later be published to [crates.io](https://crates.io) as
a standalone library, or reused by a `web` or `tui` crate without touching the CLI code.

### The `target/` directory

When you run `cargo build` or `cargo run`, Rust compiles everything into `target/`. This
directory is gitignored — it can be regenerated at any time and can grow large (several GB
on bigger projects).
