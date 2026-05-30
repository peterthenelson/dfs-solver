# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
# Build everything
cargo build

# Run all tests (across both crates)
cargo test

# Run tests for a single crate
cargo test -p variant_sudoku
cargo test -p variant_sudoku_puzzles

# Run a single test by name
cargo test test_nine_solve

# Run a specific puzzle binary (interactive TUI or CLI)
cargo run --bin double-down
cargo run --bin double-down -- --interactive
cargo run --bin double-down -- --sample_secs=10
cargo run --bin double-down -- --sample_every=10000

# Benchmarks (in variant-sudoku)
cargo run --bin bench-kropki
cargo run --bin bench-whisper
cargo run --bin diff-bench
cargo run --bin diff-stat
```

## Architecture

This is a Rust workspace with two crates:
- `variant-sudoku` — the core solver library
- `variant-sudoku-puzzles` — puzzle binaries using the library

### Core abstractions (`variant-sudoku/src/core.rs`)

The solver is generic over four orthogonal dimensions:

- **`Value`** — a finite set of values a cell can hold (e.g., `StdVal<1,9>` for digits 1–9). Internally stored as a compact unsigned integer via `UVal`.
- **`Overlay`** — defines grid structure: dimensions, region layers (rows/cols/boxes/custom). `StdOverlay<N,M>` is the standard rectangular grid with boxes.
- **`State<V, O>`** — the puzzle grid. Tracks both givens (clues) and solver-placed values separately. `apply`/`undo` are the only mutation methods; `reset()` clears placed values but keeps givens.
- **`Stateful<V>`** — `apply`/`undo`/`reset` hooks. Constraints implement this to maintain incremental state across the search.

Data structures:
- **`VBitSet<V>`** / **`VSet<V>`** — typed bitsets of `Value`s.
- **`VDenseMap<K, V>`** / **`VMap<K, V>`** — dense maps from `Value` to some type, backed by a `Box<[V]>`.
- **`Key<KT>`** — a typed interned string key; used for `Attribution` (why a contradiction occurred), `Feature` (ranker scoring), and `RegionLayer` (ROWS, COLS, BOXES, or custom).

### Constraint system (`constraint.rs`)

**`Constraint<V, O>`** (requires `Stateful<V> + Debug`) has one key method:

```rust
fn check(&self, puzzle: &State<V, O>, ranking: &mut RankingInfo<V>) -> ConstraintResult<V>
```

Returns one of: `Contradiction(Key<Attribution>)`, `Certainty(CertainDecision<V>, Key<Attribution>)`, or `Ok`. The contract: constraints should also narrow the `DecisionGrid` inside `RankingInfo` to eliminate impossible values, which makes the ranker more effective.

**`IllegalMove<V>`** is a helper used inside `Stateful::apply` implementations when an illegal placement is detected — it stores the violation to be surfaced as a `Contradiction` on the next `check()` call.

Combinators: `ConstraintConjunction<X, Y>` (two constraints, stops at first non-Ok), `MultiConstraint` (dynamic list of boxed constraints via `vec_box::vec_box![]`).

### Ranker (`ranker.rs`)

**`Ranker<V, O>`** decides *which cell or value to branch on next*. It is given a `RankingInfo<V>` (a `DecisionGrid` already partially narrowed by constraints) and returns a `BranchPoint<V>`.

**`StdRanker<O>`** uses a feature-vector / linear-scoring approach:
- Each cell gets a `FeatureVec` (keyed by `Key<Feature>`, default: `DG_CELL_POSSIBLE` = number of remaining possibilities, weight -10).
- Region info can also generate per-value feature vectors (`DG_VAL_POSSIBLE`).
- Puzzle-specific constraints add extra features (e.g., `XSUM_TAIL_FEATURE`, `CAGE_FEATURE`) to guide branching.
- `StdRanker::with_additional_weights(fv)` extends the defaults with puzzle-specific weights.

`RankingInfo<V>` wraps a `DecisionGrid<V, Raw>`. Constraints call `ranking.cells_mut()` to intersect bitsets and add features. The ranker then scores cells and values, picking the branch point with the highest score (most negative = fewest options → forced first).

### DFS solver (`solver.rs`)

**`DfsSolver<V, O, R, C>`** implements the core DFS loop. States:
- `Initializing` → replaying givens
- `Advancing` → taking a new branch
- `Backtracking` → undoing and trying alternatives
- `Solved` / `Exhausted` / `InitializationFailed`

Each step: apply a `BranchPoint`, call `check_and_rank()` (which calls `constraint.check()` then `ranker.rank()`), and decide whether to advance or backtrack.

**Higher-level APIs:**
- `FindFirstSolution` — run `solve()` to completion; returns `Option<&dyn DfsSolverView>`.
- `FindAllSolutions` — run `solve_all()` to exhaustion; use `step()` in a loop with a `StepObserver` to collect solutions.
- `PuzzleReplay` (test-util) — replay a pre-filled grid against a constraint.

**`PuzzleSetter`** is the trait every puzzle implements. It declares associated types `Value`, `Overlay`, `Ranker`, `Constraint` and implements `setup()` / `setup_with_givens()`.

### Standard Sudoku types (`sudoku.rs`)

- `StdVal<MIN, MAX>` — a `Value` for any range; `NineStdVal = StdVal<1,9>`.
- `StdOverlay<N, M>` — a `NxM` grid with configurable box layout. `nine_standard_overlay()` etc. are convenience constructors.
- `StdChecker<N, M, MIN, MAX>` — enforces standard row/col/box uniqueness. Uses `IllegalMove` internally and tracks remaining-values bitsets per row/col/box for fast narrowing.
- Type aliases: `NineStd = State<NineStdVal, NineStdOverlay>`, `EightStd`, `SixStd`, `FourStd`.

### Variant constraint modules

Each module implements `Constraint<V, O>` (and `Stateful<V>`):
- `cages` — killer/cage sums (`CageChecker`, `CageBuilder`)
- `kropki` — Kropki dot constraints
- `thermos` — thermometer constraints
- `whispers` / `dutch_whispers` — German/Dutch whispers lines
- `xsums` — X-Sum clues
- `magic_squares` — magic square constraints
- `numbered_rooms` — Numbered Rooms
- `parity_shading` — parity constraints
- `region_constraint` — generic region-based constraint
- `simple_constraint` — single-predicate constraints
- `irregular` — irregular box layouts

### TUI and puzzle entry points (`tui.rs`, `tui_std.rs`, `tui_util.rs`)

**`solve_main<P: PuzzleSetter, T: Tui<P>>()`** is the standard `main()` body for puzzle binaries. It parses CLI flags (`--interactive`, `--sample_secs=N`, `--sample_every=N`), running either an interactive ratatui UI or a headless CLI solver.

**`NineStdTui<P>`** (and `Eight/Six/FourStdTui<P>`) in `tui_std.rs` are the standard grid visualizations for `StdOverlay`-based puzzles.

**`DbgObserver`** (`debug.rs`) is a `StepObserver` that samples progress, prints stats, and writes PNG charts + JSON stats to `stats/`.

### Adding a new puzzle

1. Create `variant-sudoku-puzzles/src/bin/<name>.rs`
2. Implement `PuzzleSetter` for a new struct, defining the constraint composition with `MultiConstraint` and optionally custom ranker weights via `StdRanker::with_additional_weights`
3. Call `solve_main::<MyPuzzle, NineStdTui<_>>()` in `main()`
4. Add a `test` module with a `solve_with_given` test

### Adding a new constraint

1. Create `variant-sudoku/src/<name>.rs` and add `pub mod <name>;` to `lib.rs`
2. Implement `Stateful<V>` (for incremental state) and `Constraint<V, O>`
3. In `check()`, narrow `ranking.cells_mut()` by removing impossible values from bitsets, and add relevant features to guide the ranker
4. Use `IllegalMove<V>` if an illegal move can be detected during `apply()` but needs to be reported as a `Contradiction` during `check()`
