# Backlog

## Extant TODOs

**`ThermoChecker` stateful tracking** (`thermos.rs:170,179`)
The `Stateful` methods are effectively no-ops. Every `check()` call recomputes thermo ranges from the grid from scratch via `Range::from_set()`. Maintaining filled values per thermo incrementally in `apply`/`undo` would let you skip re-examining unchanged thermos, and would enable immediate contradiction detection in `apply` (the third TODO at `thermos.rs:170`). Probably the biggest single performance win available.

**Ranker double-pass** (`ranker.rs:590`)
`rank()` calls `to_constraint_result()` after scoring, which re-iterates over cells and region infos already visited during scoring. Combining into one pass eliminates redundant `region_info()` construction.

**TUI breakpoints** (`tui_util.rs:154`)
The readme panel says "force backtrack, breakpoints" is missing. A "run until contradiction at attribution X" breakpoint or a "run to step N" input would make the debugger significantly more useful for understanding why a constraint fires.

**Custom regions in TUI** (`tui_util.rs:4`)
The heatmap grid modes hard-code `ViewBy::{Cell,Row,Col,Box}`. Custom region layers (cage layer, kropki layer, etc.) can't be browsed. Adding `ViewBy::Custom(Key<RegionLayer>)` and a hotkey to cycle through custom layers would close this.

---

## Cleanups

**Whisper abstraction** (`whispers.rs:102`)
`GermanWhispersChecker` and `DutchWhispersChecker` share the `whisper_neighbors` table and essentially the same propagation loop, differing only in minimum distance (5 vs 4). A generic `WhispersChecker<const DIST: u8>` would collapse them.

**`ThermoChecker::debug_at`** (`thermos.rs:212`)
Currently returns nothing. Each thermo already has a color from the planar-graph coloring; at minimum, `debug_at` could report which thermos include the focused cell and their current `Range` values.

**Normalizing feature weights for region vs cell ranking** (`ranker.rs:375`)
The comment notes that choosing a `ValueInRegion` branch is implicitly over-weighted compared to a `Cell` branch because feature values aren't divided by the number of alternatives. Subtle but affects heuristic quality on heavily constrained puzzles.

---

## New constraint types

All of these are common in variant sudoku and would follow the existing `Constraint` + `Stateful` pattern.

**Arrow** — sum of digits along an arrow equals the digit in the circle. The shaft is an ordered list of cells. A stateful implementation can maintain a partial sum and propagate via the cage sum utilities already in `sudoku.rs`.

**Renban** — a line contains a set of consecutive digits in any order. Propagation: the set of possible values for each cell is the intersection of all sets of `len` consecutive digits consistent with already-placed values.

**Sandwich** — sum of digits strictly between 1 and 9 in each row/column. Compact to implement using the sum-table utilities (`stdval_sum_bound`, `stdval_len_bound`) already in `sudoku.rs`.

**Little Killer** — diagonal sum clues pointing into the grid from outside. Uses the same cage-like sum logic.

**XV / edge sums** — adjacent cells must sum to X (10) or V (5). Fits the edge-based pattern already present in Kropki.

**Diagonal** — the two main diagonals each contain digits 1–9 once. This is just adding two custom `RegionLayer`s to the overlay; the infrastructure already supports this in `StdOverlay`.

**Palindrome** — digits on a line read the same forwards and backwards. Propagation is bidirectional intersection between position `i` from the front and `i` from the back.

---

## More puzzles

Several constraints have no puzzle binaries exercising them end-to-end: magic squares, numbered rooms, parity shading, irregular boxes. Adding at least one puzzle each would exercise those code paths and provide regression test cases.

---

## Ambitious expansions

**Solution count / uniqueness checker**
`FindAllSolutions` exists but there's no convenience wrapper for "does this puzzle have exactly one solution?" — useful for puzzle construction. A `SolutionCount` helper with early exit after finding a second solution would be small but valuable.

**Puzzle generation**
Given a constraint set, iteratively place givens and check uniqueness to generate a valid puzzle. The seeded `ChaCha20Rng` in `bench.rs` suggests this direction has been considered.

**Construction mode in TUI**
A TUI mode where you interactively place constraints and export a `PuzzleSetter` struct (or JSON representation) would let you build puzzles without hand-editing source. Builds on the existing `solve_interactive` entry point and the manual-entry modal infrastructure.
