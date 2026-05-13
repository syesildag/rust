# Selfplay Batch Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace N sequential `model.forward()` calls per ply with a single `model.forward_batch()` call, eliminating the `eval_move` helper.

**Architecture:** All changes are confined to `crates/engine/src/selfplay.rs`. The selected code block (lines 101–106) is replaced by: collect resulting boards into a `Vec<Board>`, call `forward_batch` once, apply a sign flip for Black, then pick the argmax. `eval_move` is deleted.

**Tech Stack:** Rust, `crates/engine`, `HybridValueNet::forward_batch` (already implemented in `crates/engine/src/model.rs`).

---

### Task 1: Regression guard test

This is a refactor — behaviour must not change. Add the test first so it acts as a safety net during implementation.

**Files:**
- Modify: `crates/engine/src/selfplay.rs`

- [ ] **Step 1: Add the regression test at the bottom of `selfplay.rs`**

Append this block to the end of `crates/engine/src/selfplay.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::HybridValueNet;

    #[test]
    fn generate_one_game_produces_samples() {
        let model = HybridValueNet::new();
        let dataset = generate(&model, 1);
        assert!(!dataset.is_empty(), "expected at least one training sample");
    }
}
```

- [ ] **Step 2: Run the test to confirm it passes on the current code**

```bash
cargo test -p engine selfplay::tests::generate_one_game_produces_samples
```

Expected: `test selfplay::tests::generate_one_game_produces_samples ... ok`

---

### Task 2: Implement batch move selection

Replace the `max_by` block and delete `eval_move`.

**Files:**
- Modify: `crates/engine/src/selfplay.rs:101-142`

- [ ] **Step 1: Replace lines 101–106 (the `max_by` block) with the batch path**

Old code (lines 101–106):
```rust
        // Pick the move maximising value from the side-to-move's perspective.
        let best_move = legal.iter().copied().max_by(|&a, &b| {
            let va = eval_move(model, &board, a);
            let vb = eval_move(model, &board, b);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });
```

Replace with:
```rust
        // Evaluate all legal moves in one batched forward pass.
        let after_boards: Vec<Board> = legal.iter().copied().map(|mv| board.make_move(mv)).collect();
        let raw_data = model.forward_batch(&after_boards).data();
        let sign = match board.side_to_move {
            Color::White => 1.0_f32,
            Color::Black => -1.0_f32,
        };
        let best_move = (0..legal.len())
            .max_by(|&i, &j| {
                (sign * raw_data[i])
                    .partial_cmp(&(sign * raw_data[j]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|i| legal[i]);
```

- [ ] **Step 2: Delete the `eval_move` function (lines 132–142)**

Remove this entire function from `selfplay.rs`:
```rust
/// Evaluates a candidate move by running the model on the resulting position,
/// negating for Black (so higher is always better for the side to move).
fn eval_move(model: &HybridValueNet, board: &Board, mv: chess::moves::Move) -> f32 {
    let after = board.make_move(mv);
    let raw = model.forward(&after).data()[0];
    // White maximises positive values; Black maximises negative (flips sign).
    match board.side_to_move {
        Color::White => raw,
        Color::Black => -raw,
    }
}
```

- [ ] **Step 3: Compile to confirm no errors**

```bash
cargo check -p engine
```

Expected: `Finished` with no errors or warnings (pedantic clippy is not run here, just type-check).

---

### Task 3: Verify

- [ ] **Step 1: Run the regression test**

```bash
cargo test -p engine selfplay::tests::generate_one_game_produces_samples
```

Expected: `test selfplay::tests::generate_one_game_produces_samples ... ok`

- [ ] **Step 2: Run the full lint suite**

```bash
cargo check-all
```

Expected: exit 0, no warnings.

- [ ] **Step 3: Run all engine tests**

```bash
cargo test -p engine
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/engine/src/selfplay.rs
git commit -m "perf(selfplay): replace per-move forward() with forward_batch()"
```
