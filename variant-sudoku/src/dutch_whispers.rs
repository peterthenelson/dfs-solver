use std::collections::HashMap;
use std::fmt::Debug;
use crate::constraint::Constraint;
use crate::core::{full_set, singleton_set, unpack_singleton, Attribution, ConstraintResult, DecisionGrid, Error, FeatureKey, Index, Overlay, State, Stateful, UVSet, Value, WithId};
use crate::index_util::expand_polyline;
use crate::sudoku::{unpack_stdval_vals, NineStdVal, StdOverlay, StdState};
use crate::whispers::{whisper_between, whisper_neighbors, whisper_possible_values};

/// DutchWhispers are a line-based constraint where adjacent cells on the line
/// must have (normal sudoku 1-9) values at least 4 apart. There are other
/// line-based "whisper" constraints, but this one is common enough to warrant
/// a specialized implementation.
#[derive(Debug, Clone)]
pub struct DutchWhisper {
    // Cell in the whisper and whether it has two mutually visible neighbors.
    pub cells: Vec<(Index, bool)>,
}

impl DutchWhisper {
    pub fn contains(&self, index: Index) -> bool {
        self.cells.iter().any(|(i, _)| *i == index)
    }
}

// DutchWhisperConstraint works more efficiently when the mutual-visibility of
// neighbors is known, but it's inconvenient to manually specify that.
pub struct DutchWhisperBuilder<'a, O: Overlay>(&'a O);
impl <'a, O: Overlay> DutchWhisperBuilder<'a, O> {
    pub fn new(visibility_constraint: &'a O) -> Self {
        Self(visibility_constraint)
    }

    pub fn whisper(&self, cells: Vec<Index>) -> DutchWhisper {
        let mut has_mutual_visibility = vec![false; cells.len()];
        for (i, &cell) in cells.iter().enumerate() {
            if i > 0 {
                let prev = cells[i - 1];
                let diff = (cell[0].abs_diff(prev[0]), cell[1].abs_diff(prev[1]));
                if diff != (0, 1) && diff != (1, 0) && diff != (1, 1) {
                    panic!("Cells {:?} and {:?} are not adjacent", cell, prev);
                }
            }
            if i < cells.len() - 1{
                let next = cells[i + 1];
                let diff = (cell[0].abs_diff(next[0]), cell[1].abs_diff(next[1]));
                if diff != (0, 1) && diff != (1, 0) && diff != (1, 1) {
                    panic!("Cells {:?} and {:?} are not adjacent", cell, next);
                }
            }
            if i > 0 && i < cells.len() - 1 {
                let prev = cells[i - 1];
                let next = cells[i + 1];
                has_mutual_visibility[i] = self.0.mutually_visible(prev, next);
            }
        }
        DutchWhisper {
            cells: cells.into_iter().zip(has_mutual_visibility).collect(),
        }
    }

    pub fn row(&self, left: Index, length: usize) -> DutchWhisper {
        let cells = (0..length)
            .map(|i| [left[0], left[1] + i])
            .collect();
        self.whisper(cells)
    }

    pub fn col(&self, top: Index, length: usize) -> DutchWhisper {
        let cells = (0..length)
            .map(|i| [top[0] + i, top[1]])
            .collect();
        self.whisper(cells)
    }

    pub fn polyline(&self, vertices: Vec<Index>) -> DutchWhisper {
        self.whisper(expand_polyline(vertices).unwrap())
    }
}

pub const DW_ILLEGAL_ACTION: Error = Error::new_const("A dutch-whisper violation already exists; can't apply further actions.");
pub const DW_UNDO_MISMATCH: Error = Error::new_const("Undo value mismatch");
pub const DW_FEATURE: &str = "DUTCH_WHISPER";
pub const DW_TOO_CLOSE_ATTRIBUTION: &str = "DW_TOO_CLOSE";

pub struct DutchWhisperChecker {
    whispers: Vec<DutchWhisper>,
    remaining_init: HashMap<Index, UVSet<u8>>,
    remaining: HashMap<Index, UVSet<u8>>,
    dw_feature: FeatureKey<WithId>,
    dw_too_close_attribution: Attribution<WithId>,
    illegal: Option<(Index, NineStdVal, Attribution<WithId>)>,
}

impl DutchWhisperChecker {
    pub fn new(whispers: Vec<DutchWhisper>) -> Self {
        let mut remaining = HashMap::new();
        for w in &whispers {
            for (cell, h2mvn) in &w.cells {
                remaining.entry(*cell).or_insert_with(|| {
                    whisper_possible_values::<1, 9>(4, *h2mvn)
                });
            }
        }
        Self {
            whispers,
            remaining_init: remaining.clone(),
            remaining,
            dw_feature: FeatureKey::new(DW_FEATURE).unwrap(),
            dw_too_close_attribution: Attribution::new(DW_TOO_CLOSE_ATTRIBUTION).unwrap(),
            illegal: None,
        }
    }
}

impl Debug for DutchWhisperChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, w) in self.whispers.iter().enumerate() {
            write!(f, " Whisper[{}]: ", i)?;
            for (cell, _) in &w.cells {
                let rem = self.remaining.get(cell)
                    .expect(format!("remaining[{:?}] not found!", cell).as_str());
                write!(f, "{:?}=>{:?} ", cell, unpack_stdval_vals::<1, 9>(rem))?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

fn recompute(remaining: &mut HashMap<Index, UVSet<u8>>, remaining_init: &HashMap<Index, UVSet<u8>>, w: &DutchWhisper, i: usize) {
    let mut rem = remaining_init.get(&w.cells[i].0).unwrap().clone();
    if i > 0 {
        let prev = w.cells[i - 1].0;
        let prev_rem = remaining.get(&prev).unwrap().clone();
        if let Some(v) = unpack_singleton::<NineStdVal>(&prev_rem) {
            rem.intersect_with(&whisper_neighbors::<1, 9>(4, v));
        }
    }
    if i < w.cells.len() - 1 {
        let next = w.cells[i + 1].0;
        let next_rem = remaining.get(&next).unwrap().clone();
        if let Some(v) = unpack_singleton::<NineStdVal>(&next_rem) {
            rem.intersect_with(&whisper_neighbors::<1, 9>(4, v));
        }
    }
    *remaining.get_mut(&w.cells[i].0).unwrap() = rem;
}

impl Stateful<NineStdVal> for DutchWhisperChecker {
    fn reset(&mut self) {
        self.remaining = self.remaining_init.clone();
        self.illegal = None;
    }

    fn apply(&mut self, index: Index, value: NineStdVal) -> Result<(), Error> {
        // In theory we could be allow multiple illegal moves and just
        // invalidate and recalculate the grid or something, but it seems hard.
        if self.illegal.is_some() {
            return Err(DW_ILLEGAL_ACTION);
        }
        if !self.remaining.contains_key(&index) {
            return Ok(());
        }
        for w in &self.whispers {
            for (i, (cell, _)) in w.cells.iter().enumerate() {
                if *cell != index {
                    continue;
                }
                let neighbors = whisper_neighbors::<1, 9>(4, value);
                let cur = self.remaining.get_mut(&index).unwrap();
                if cur.contains(value.to_uval()) {
                    *cur = singleton_set::<NineStdVal>(value);
                } else {
                    self.illegal = Some((index, value, self.dw_too_close_attribution.clone()));
                    return Ok(());
                }
                if i > 0 {
                    let prev = w.cells[i - 1].0;
                    self.remaining.get_mut(&prev).unwrap().intersect_with(&neighbors);
                }
                if i < w.cells.len() - 1 {
                    let next = w.cells[i + 1].0;
                    self.remaining.get_mut(&next).unwrap().intersect_with(&neighbors);
                }
            }
        }
        Ok(())
    }

    fn undo(&mut self, index: Index, value: NineStdVal) -> Result<(), Error> {
        if let Some((i, v, _)) = self.illegal {
            if i != index || v != value {
                return Err(DW_UNDO_MISMATCH);
            } else {
                self.illegal = None;
                return Ok(());
            }
        }
        if !self.remaining.contains_key(&index) {
            return Ok(());
        }
        for w in &self.whispers {
            for (i, (cell, _)) in w.cells.iter().enumerate() {
                if *cell != index {
                    continue;
                }
                recompute(&mut self.remaining, &self.remaining_init, w, i);
            }
        }
        Ok(())
    }
}

impl <const N: usize, const M: usize>
Constraint<NineStdVal, StdOverlay<N, M>, StdState<N, M, 1, 9>> for DutchWhisperChecker {
    fn check(&self, puzzle: &StdState<N, M, 1, 9>, grid: &mut DecisionGrid<NineStdVal>) -> ConstraintResult<NineStdVal> {
        if let Some((_, _, a)) = &self.illegal {
            return ConstraintResult::Contradiction(a.clone());
        }
        for w in &self.whispers {
            for (cell, _) in w.cells.iter() {
                if puzzle.get(*cell).is_some() {
                    continue;
                }
                let g = grid.get_mut(*cell);
                g.0.intersect_with(&self.remaining.get(cell).unwrap());
                g.1.add(&self.dw_feature, 1.0);
            }
        }
        for w in &self.whispers {
            for (i, (cell, _)) in w.cells.iter().enumerate() {
                if puzzle.get(*cell).is_some() {
                    continue;
                }
                let left = if i > 0 {
                    let prev = w.cells[i - 1].0;
                    let mut prev_set = self.remaining.get(&prev).unwrap().clone();
                    if puzzle.get(prev).is_none() {
                        prev_set.intersect_with(&grid.get(prev).0);
                    }
                    prev_set
                } else {
                    full_set::<NineStdVal>()
                };
                let right = if i < w.cells.len() - 1 {
                    let next = w.cells[i + 1].0;
                    let mut next_set = self.remaining.get(&next).unwrap().clone();
                    if puzzle.get(next).is_none() {
                        next_set.intersect_with(&grid.get(next).0);
                    }
                    next_set
                } else {
                    full_set::<NineStdVal>()
                };
                grid.get_mut(*cell).0.intersect_with(
                    &whisper_between::<1, 9>(4, &left, &right),
                );
            }
        }
        ConstraintResult::Ok
    }

    fn debug_at(&self, _: &StdState<N, M, 1, 9>, index: Index) -> Option<String> {
        let header = "DutchWhisperChecker:\n";
        let mut lines = vec![];
        for (i, w) in self.whispers.iter().enumerate() {
            if !w.contains(index) {
                continue;
            }
            lines.push(format!("  Whisper[{}]: ", i));
            for (cell, _) in &w.cells {
                if *cell != index {
                    continue;
                }
                let rem = self.remaining.get(cell)
                    .expect(format!("remaining[{:?}] not found!", cell).as_str());
                lines.push(format!("  - remaining vals: {:?}", unpack_stdval_vals::<1, 9>(rem)));
            }
        }
        if lines.is_empty() {
            None
        } else {
            Some(format!("{}{}", header, lines.join("\n")))
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{constraint::{test_util::assert_contradiction, MultiConstraint}, ranker::StdRanker, solver::test_util::PuzzleReplay, sudoku::{nine_standard_overlay, nine_standard_parse, StdChecker}};
    use super::*;

    #[test]
    fn test_dutch_whisper_builder() {
        let overlay = nine_standard_overlay();
        assert!(overlay.mutually_visible([4, 2], [5, 0]));
        let dw = DutchWhisperBuilder::new(&overlay);
        assert_eq!(
            dw.whisper(vec![[2, 1], [2, 2], [3, 3], [4, 2], [5, 1], [5, 0]]).cells,
            vec![
                ([2, 1], false),  // At the beginning, doesn't have 2 neighbors
                ([2, 2], false),  // [2, 1] and [3, 3] don't see each other
                ([3, 3], true),   // [2, 2] and [4, 2] do see each other
                ([4, 2], false),  // [3, 3] and [5, 1] don't see each other
                ([5, 1], true),   // [4, 2] and [5, 0] do see each other
                ([5, 0], false),  // At the end, doesn't have 2 neighbors
            ],
        );
    }

    #[test]
    #[should_panic]
    fn test_dutch_whisper_builder_nonadjacent() {
        let overlay = nine_standard_overlay();
        let dw = DutchWhisperBuilder::new(&overlay);
        // [2, 2] <-> [2, 4] are not adjacent, even diagonally
        dw.whisper(vec![[2, 1], [2, 2], [2, 4]]);
    }

    // TODO: more detailed testing of the constraint.
    #[test]
    fn test_dutch_whisper_constraint() {
        // This is a 9x9 puzzle with a whisper going from [0, 0] over to [0, 4]
        // and down to [4, 4]. I show how different initial setups lead to
        // contradiction or not.
        let overlay = nine_standard_overlay();
        let db = DutchWhisperBuilder::new(&overlay);
        let whispers = vec![db.polyline(vec![[0, 0], [0, 4], [4, 4]])];
        let ranker = StdRanker::default();
        for (attr, bad_setup) in [
            (
                // 8 and 5 are too close
                "DW_TOO_CLOSE",
                "85.......\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n",
            ),
            (
                // [0, 2] must be a 1 for DWs reasons, but it's ruled out for
                // Sudoku reasons.
                "DG_CELL_NO_VALS",
                "95......1\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n",
            ),
            (
                // [0, 4] is squeezed between 8 and 2 -- there are not values
                // four from both.
                "DG_CELL_NO_VALS",
                "...8....1\n\
                 ....2....\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n\
                 .........\n",
            ),
        ] {
            let mut puzzle = nine_standard_parse(bad_setup).unwrap();
            let mut constraint = MultiConstraint::new(vec_box::vec_box![
                StdChecker::new(&puzzle),
                DutchWhisperChecker::new(whispers.clone()),
            ]);
            let result = PuzzleReplay::new(&mut puzzle, &ranker, &mut constraint, None).replay().unwrap();
            assert_contradiction(result, attr);
        }
    }
}