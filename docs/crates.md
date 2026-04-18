# Crates

This workspace has two crates. Each crate has its own `Cargo.toml` manifest but inherits
shared configuration (version, edition, lint rules) from the workspace root.

---

## `core` — Library Crate

**Location:** [`crates/core/`](../crates/core/)
**Entry point:** [`crates/core/src/lib.rs`](../crates/core/src/lib.rs)

A library crate. It has no `main` function — it cannot be run directly. It exposes public
functions that other crates (like `cli`) can import and use.

### Current API

#### `add(a: i32, b: i32) -> i32`

Adds two integers and returns the result.

```rust
use core::add;

let sum = add(2, 3);  // → 5
```

The `#[must_use]` attribute on this function means the compiler will warn you if you call
`add(...)` but don't use the return value — a common accidental mistake.

### Adding New Functions

Add public functions to `lib.rs` (or create new modules inside `crates/core/src/`):

```rust
/// Multiplies two integers.
///
/// # Examples
///
/// ```
/// assert_eq!(core::multiply(3, 4), 12);
/// ```
#[must_use]
pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}
```

The `///` doc comment and the code block inside it become a doc test — `cargo test` will
compile and run it automatically.

---

## `cli` — Binary Crate

**Location:** [`crates/cli/`](../crates/cli/)
**Entry point:** [`crates/cli/src/main.rs`](../crates/cli/src/main.rs)

A binary crate. It has a `fn main()` — this is what runs when you do `cargo run -p cli`.
Its job is to wire together logic from `core` and present it to the user.

### Current Behaviour

```rust
fn main() {
    let result = core::add(6, 7);
    println!("6 + 7 = {result}");
}
```

Output:

```
6 + 7 = 13
```

### Dependency on `core`

Declared in [`crates/cli/Cargo.toml`](../crates/cli/Cargo.toml):

```toml
[dependencies]
core = { path = "../core" }
```

`path = "../core"` means "use the local crate at that path" rather than downloading from
the internet. This is how workspace crates reference each other.

---

## Adding a New Crate

To add a third crate (e.g. a `web` server):

1. Create `crates/web/src/main.rs` and `crates/web/Cargo.toml`
2. Add `"crates/web"` to the `members` list in the root `Cargo.toml`
3. Reference `core` as a dependency in `crates/web/Cargo.toml` the same way `cli` does
