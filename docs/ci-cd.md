# CI/CD

Continuous Integration (CI) automatically runs checks on every push and pull request, so
problems are caught before they reach the main branch.

Configuration: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

---

## When Does CI Run?

```yaml
on:
  push:
    branches: [main]
  pull_request:
```

- Every push to `main`
- Every pull request (on open and on new commits)

---

## Jobs

The pipeline has three independent jobs that run in parallel:

### 1. `fmt` — Formatting Check

```bash
cargo fmt --check
```

Verifies that all code is formatted according to `rustfmt.toml`. Does **not** modify files
— it exits with an error if anything is unformatted. This enforces a consistent code style
without any manual review effort.

**Fix locally:** `cargo fmt` then commit the result.

### 2. `clippy` — Lint Check

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

Runs the Clippy linter across every crate, every target (lib, bin, tests), and every
feature flag. `-D warnings` promotes all warnings to errors so nothing slips through.

**Fix locally:** `cargo lint` (the project alias).

### 3. `test` — Test Suite

```bash
cargo test --all
```

Compiles and runs all tests across all crates — unit tests, integration tests, and doc
tests.

**Fix locally:** `cargo test --all`, investigate any failures.

---

## Infrastructure

| Tool | Purpose |
|---|---|
| `dtolnay/rust-toolchain@stable` | Installs the Rust toolchain in CI (reads `rust-toolchain.toml`) |
| `Swatinem/rust-cache@v2` | Caches compiled dependencies between CI runs — speeds up builds significantly |
| `CARGO_TERM_COLOR: always` | Forces coloured output in CI logs for readability |

---

## Pre-CI Local Check

Before pushing, run:

```bash
cargo check-all
```

This alias runs `cargo fmt --check && cargo lint` — the same checks CI runs for `fmt` and
`clippy` jobs, locally, in seconds. Catching issues before push saves a CI round-trip.
