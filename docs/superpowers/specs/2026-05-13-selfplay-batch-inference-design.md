# Self-play Batch Inference Design

**Date:** 2026-05-13
**Status:** Approved

## Problem

`selfplay.rs` evaluates each legal move by calling `model.forward()` individually inside a `max_by` comparator. For a typical chess position with ~30 legal moves, this means 30 separate neural-network forward passes per ply. This is the dominant per-ply cost and the primary bottleneck for move-selection speed.

## Goal

Replace N sequential `forward` calls with a single `forward_batch` call per ply, keeping all existing interfaces and behaviour unchanged.

## Scope

- **In scope:** `crates/engine/src/selfplay.rs` only.
- **Out of scope:** parallel game generation, changes to the tensor or model crates, changes to training or PGN code.

## Design

### Data flow

```
legal: Vec<Move>  (N moves, typically ~30)
  → sequential map(make_move) → after_boards: Vec<Board>
  → model.forward_batch(&after_boards) → scores: Tensor [N, 1]
  → sign-flip for Black (negate all scores when side_to_move == Black)
  → argmax over N scores → best_idx
  → legal[best_idx] → best_move: Option<Move>
```

### Board construction

`board.make_move(mv)` for each legal move is collected into a `Vec<Board>` sequentially. This operation is a pure bitboard transform (~10 ns per call). With ~30 moves, rayon dispatch overhead (~200–500 ns per task) would exceed the work itself, making parallelism counter-productive at this granularity. Sequential collection is correct.

### Batch inference

`HybridValueNet::forward_batch` is already implemented and fully vectorised (CLS prepend, positional embeddings, transformer encoder, and CLS extraction all operate on the full batch). One call replaces N calls, with better SIMD utilisation and amortised overhead.

### Sign normalisation

`eval_move` today returns `raw` for White and `-raw` for Black so that "higher is always better for the side to move." In the batch path, the sign flip is applied to the entire `Tensor [N, 1]` after `forward_batch` returns, before argmax. This is identical semantics, no per-item branching needed.

### Argmax

Iterate once over the `N` scores from the normalised tensor, tracking the index of the maximum. Index into `legal` to recover the `Move`. Returns `None` only when `legal` is empty (same as the existing `max_by` path).

### Deletion

`eval_move` is deleted. Its logic is fully absorbed by the batch pipeline.

## Interfaces unchanged

| Symbol | Change |
|---|---|
| `generate(model, num_games) -> ChessDataset` | None |
| `generate_with_pgn(model, num_games)` | None |
| `play_game(model) -> PlayedGame` | Internal only — body changes |
| `eval_move` | Deleted |

## Trade-offs considered

| Approach | Verdict |
|---|---|
| Rayon `par_iter` over moves | Rejected — dispatch overhead dominates for ~30 items; doesn't eliminate N passes |
| Thread-local model copies | Rejected — requires deep-copy path (serialize/deserialize); disproportionate complexity |
| `forward_batch` (chosen) | Best: infrastructure already exists, zero thread-safety changes, N→1 forward passes |
