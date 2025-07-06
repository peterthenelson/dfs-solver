use std::collections::HashMap;
use std::fmt::Debug;
use crate::color_util::{color_ave, color_polarity};
use crate::constraint::Constraint;
use crate::core::{Attribution, ConstraintResult, Error, Feature, Index, Key, Overlay, State, Stateful, VBitSet, VSet, VSetMut};
use crate::index_util::{check_adjacent, expand_polyline};
use crate::ranker::RankingInfo;
use crate::sudoku::{unpack_stdval_vals, NineStdVal, StdOverlay};
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
                check_adjacent(cells[i - 1], cell).unwrap();
            }
            if i > 0 && i < cells.len() - 1 {
                has_mutual_visibility[i] = self.0.mutually_visible(cells[i - 1], cells[i + 1]);
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
    remaining_init: HashMap<Index, VBitSet<NineStdVal>>,
    remaining: HashMap<Index, VBitSet<NineStdVal>>,
    dw_feature: Key<Feature>,
    dw_too_close_attr: Key<Attribution>,
    illegal: Option<(Index, NineStdVal, Key<Attribution>)>,
}

impl DutchWhisperChecker {
    pub fn new(whispers: Vec<DutchWhisper>) -> Self {
        let mut remaining = HashMap::new();
        for w in &whispers {
            for (cell, h2mvn) in &w.cells {
                remaining.entry(*cell).or_insert_with(|| {
                    whisper_possible_values::<1, 9>().get(4, *h2mvn).to_vbitset()
                });
            }
        }
        Self {
            whispers,
            remaining_init: remaining.clone(),
            remaining,
            dw_feature: Key::register(DW_FEATURE),
            dw_too_close_attr: Key::register(DW_TOO_CLOSE_ATTRIBUTION),
            illegal: None,
        }
    }
}

impl Debug for DutchWhisperChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((i, v, a)) = &self.illegal {
            write!(f, "Illegal move: {:?}={:?} ({})\n", i, v, a.name())?;
        }
        for (i, w) in self.whispers.iter().enumerate() {
            write!(f, " Whisper[{}]: ", i)?;
            for (cell, _) in &w.cells {
                let rem = self.remaining.get(cell)
                    .expect(format!("remaining[{:?}] not found!", cell).as_str());
                write!(f, "{:?}=>{:?} ", cell, unpack_stdval_vals::<1, 9, _>(rem))?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

fn recompute(remaining: &mut HashMap<Index, VBitSet<NineStdVal>>, remaining_init: &HashMap<Index, VBitSet<NineStdVal>>, w: &DutchWhisper, i: usize) {
    let mut rem = remaining_init.get(&w.cells[i].0).unwrap().clone();
    if i > 0 {
        let prev = w.cells[i - 1].0;
        let prev_rem = remaining.get(&prev).unwrap().clone();
        if let Some(v) = prev_rem.as_singleton() {
            rem.intersect_with(&whisper_neighbors::<1, 9>().get(4, v));
        }
    }
    if i < w.cells.len() - 1 {
        let next = w.cells[i + 1].0;
        let next_rem = remaining.get(&next).unwrap().clone();
        if let Some(v) = next_rem.as_singleton() {
            rem.intersect_with(&whisper_neighbors::<1, 9>().get(4, v));
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
                let mut wn = whisper_neighbors::<1, 9>();
                let neighbors = wn.get(4, value);
                let cur = self.remaining.get_mut(&index).unwrap();
                if cur.contains(&value) {
                    *cur = VBitSet::<NineStdVal>::singleton(&value);
                } else {
                    self.illegal = Some((index, value, self.dw_too_close_attr));
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
Constraint<NineStdVal, StdOverlay<N, M>> for DutchWhisperChecker {
    fn name(&self) -> Option<String> { Some("DutchWhisperChecker".to_string()) }
    fn check(&self, puzzle: &State<NineStdVal, StdOverlay<N, M>>, ranking: &mut RankingInfo<NineStdVal>) -> ConstraintResult<NineStdVal> {
        if let Some((_, _, a)) = &self.illegal {
            return ConstraintResult::Contradiction(*a);
        }
        let grid = ranking.cells_mut();
        for w in &self.whispers {
            for (cell, _) in w.cells.iter() {
                if puzzle.get(*cell).is_some() {
                    continue;
                }
                let g = grid.get_mut(*cell);
                g.0.intersect_with(self.remaining.get(cell).unwrap());
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
                        prev_set.intersect_with(grid.get(prev).0);
                    }
                    prev_set
                } else {
                    VBitSet::<NineStdVal>::full()
                };
                let right = if i < w.cells.len() - 1 {
                    let next = w.cells[i + 1].0;
                    let mut next_set = self.remaining.get(&next).unwrap().clone();
                    if puzzle.get(next).is_none() {
                        next_set.intersect_with(grid.get(next).0);
                    }
                    next_set
                } else {
                    VBitSet::<NineStdVal>::full()
                };
                grid.get_mut(*cell).0.intersect_with(
                    &whisper_between::<1, 9, _>(4, &left, &right),
                );
            }
        }
        ConstraintResult::Ok
    }

    fn debug_at(&self, _: &State<NineStdVal, StdOverlay<N, M>>, index: Index) -> Option<String> {
        let header = "DutchWhisperChecker:\n";
        let mut lines = vec![];
        if let Some((i, v, a)) = &self.illegal {
            if *i == index {
                lines.push(format!("  Illegal move: {:?}={:?} ({})", i, v, a.name()));
            }
        }
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
                lines.push(format!("  - remaining vals: {:?}", unpack_stdval_vals::<1, 9, _>(rem)));
            }
        }
        if lines.is_empty() {
            None
        } else {
            Some(format!("{}{}", header, lines.join("\n")))
        }
    }

    fn debug_highlight(&self, puzzle: &State<NineStdVal, StdOverlay<N, M>>, index: Index) -> Option<(u8, u8, u8)> {
        if let Some((i, _, _)) = &self.illegal {
            if *i == index {
                return Some((200, 0, 0));
            }
        }
        if let Some(rem) = self.remaining.get(&index) {
            if let Some(v) = puzzle.get(index) {
                return Some(color_polarity(1, 9, v.val()))
            }
            let polarities = rem.iter()
                .map(|v| color_polarity(1, 9, v.val()))
                .collect::<Vec<_>>();
            Some(color_ave(&polarities))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{constraint::{test_util::{assert_contradiction, assert_no_contradiction}, MultiConstraint}, ranker::StdRanker, solver::test_util::PuzzleReplay, sudoku::{nine_standard_overlay, StdChecker}};
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

    // This is a 9x9 puzzle with a whisper going from [0, 0] over to [0, 4]
    // and down to [4, 4]. Call with different givens and an expectation for it
    // to return a contradiction (or not).
    fn assert_dutch_whisper_result(
        setup: &str, 
        expected: Option<&'static str>,
    ) {
        let overlay = nine_standard_overlay();
        let db = DutchWhisperBuilder::new(&overlay);
        let whispers = vec![db.polyline(vec![[0, 0], [0, 4], [4, 4]])];
        let ranker = StdRanker::default();
        let mut puzzle = overlay.parse_state::<NineStdVal>(setup).unwrap();
        let mut constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(&overlay),
            DutchWhisperChecker::new(whispers),
        ]);
        let result = PuzzleReplay::new(&mut puzzle, &ranker, &mut constraint, None).replay().unwrap();
        if let Some(attr) = expected {
            assert_contradiction(result, attr);
        } else {
            assert_no_contradiction(result);
        }
    }

    #[test]
    fn test_dutch_whisper_too_close() {
        // 8 and 5 are too close
        let input: &str = "8 5 .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           -----+-----+-----\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           -----+-----+-----\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n";
        assert_dutch_whisper_result(input, Some("DW_TOO_CLOSE"));
    }

    #[test]
    fn test_dutch_whisper_sudoku_interaction() {
        // [0, 2] must be a 1 for DWs reasons, but it's ruled out for
        // Sudoku reasons.
        let input: &str = "9 5 .|. . .|. . 1\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           -----+-----+-----\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           -----+-----+-----\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n";
        assert_dutch_whisper_result(input, Some("DG_CELL_NO_VALS"));
    }

    #[test]
    fn test_dutch_whisper_squeeze() {
        // [0, 4] is squeezed between 8 and 2 -- there are not values that work
        // for both.
        let input: &str = ". . .|8 . .|. . .\n\
                           . . .|. 2 .|. . .\n\
                           . . .|. . .|. . .\n\
                           -----+-----+-----\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           -----+-----+-----\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n";
        assert_dutch_whisper_result(input, Some("DG_CELL_NO_VALS"));
    }

    #[test]
    fn test_dutch_whisper_valid_fill() {
        // Valid fill
        let input: &str = "1 5 9|2 8 .|. . .\n\
                           . . .|. 3 .|. . .\n\
                           . . .|. 7 .|. . .\n\
                           -----+-----+-----\n\
                           . . .|. 2 .|. . .\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           -----+-----+-----\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n\
                           . . .|. . .|. . .\n";
        assert_dutch_whisper_result(input, None);
    }
}