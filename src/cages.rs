use std::collections::HashSet;
use std::fmt::Debug;
use crate::core::{full_set, ConstraintResult, DecisionGrid, Error, FKWithId, FeatureKey, Index, Set, Stateful, Value};
use crate::constraint::{Constraint, ConstraintViolationDetail};
use crate::sudoku::{unpack_sval_vals, SState, SVal, Overlay};

#[derive(Debug, Clone)]
pub struct Cage {
    pub target: u8,
    pub cells: Vec<Index>,
    pub exclusive: bool,
}

impl Cage {
    pub fn contains(&self, index: Index) -> bool {
        self.cells.contains(&index)
    }
}

// Exclusive cages are more efficient for the solver. If the ruleset doesn't
// include exclusivity, individual cages may still be exclusive if all of their
// digits see each other (e.g., same column or row) according to other rules in
// the ruleset. Since "normal sudoku rules" is a common case, this inference
// can be automated by creating Cages with this function.
pub struct CageBuilder<'a, O: Overlay>(bool, &'a O);
impl <'a, O: Overlay> CageBuilder<'a, O> {
    pub fn new(exclusive: bool, visibility_constraint: &'a O) -> Self {
        Self(exclusive, visibility_constraint)
    }

    pub fn cage(&self, target: u8, cells: Vec<Index>) -> Cage {
        let exclusive = self.0 || self.1.all_mutually_visible(&cells);
        Cage { target, cells, exclusive }
    }
}

pub const ILLEGAL_ACTION_CAGE: Error = Error::new_const("A cage violation already exists; can't apply further actions.");
pub const UNDO_MISMATCH: Error = Error::new_const("Undo value mismatch");
pub const CAGE_FEATURE: &str = "CAGE";

pub struct CageChecker<const MIN: u8, const MAX: u8> {
    cages: Vec<Cage>,
    remaining: Vec<u8>,
    empty: Vec<usize>,
    cage_sets: Vec<Set<u8>>,
    cage_feature: FeatureKey<FKWithId>,
    illegal: Option<(Index, SVal<MIN, MAX>)>,
}

impl <const MIN: u8, const MAX: u8> CageChecker<MIN, MAX> {
    // Note: Cages must not overlap with each other
    pub fn new(cages: Vec<Cage>) -> Self {
        let mut covered = HashSet::new();
        for c in &cages {
            for cell in &c.cells {
                if covered.contains(&cell) {
                    panic!("Multiple cages contain cell: {:?}\n", cell);
                }
                covered.insert(cell);
            }
        }
        let remaining = cages.iter().map(|c| c.target).collect();
        let empty = cages.iter().map(|c| c.cells.len()).collect();
        let cage_sets = vec![full_set::<u8, SVal<MIN, MAX>>(); cages.len()];
        let mut cage_feature = FeatureKey::new(CAGE_FEATURE);
        CageChecker { cages, remaining, empty, cage_sets, illegal: None, cage_feature: cage_feature.unwrap() }
    }
}

impl <const MIN: u8, const MAX: u8> Debug for CageChecker<MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((i, v)) = self.illegal {
            write!(f, "Illegal move: {:?}; {:?}\n", i, v)?;
        }
        for (i, c) in self.cages.iter().enumerate() {
            write!(f, " Cage{:?}\n", c.cells)?;
            write!(f, " - {} remaining to target; {} cells empty\n", self.remaining[i], self.empty[i])?;
            if c.exclusive {
                let vals = unpack_sval_vals::<MIN, MAX>(&self.cage_sets[i]);
                write!(f, " - Unused vals: {:?}\n", vals)?;
            }
        }
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8> Stateful<u8, SVal<MIN, MAX>> for CageChecker<MIN, MAX> {
    fn reset(&mut self) {
        self.remaining = self.cages.iter().map(|c| c.target).collect();
        self.empty = self.cages.iter().map(|c| c.cells.len()).collect();
        self.cage_sets = vec![full_set::<u8, SVal<MIN, MAX>>(); self.cages.len()];
        self.illegal = None;
    }

    fn apply(&mut self, index: Index, value: SVal<MIN, MAX>) -> Result<(), Error> {
        let uv = value.to_uval();
        // In theory we could be allow multiple illegal moves and just
        // invalidate and recalculate the grid or something, but it seems hard.
        if self.illegal.is_some() {
            return Err(ILLEGAL_ACTION_CAGE);
        }
        for (i, c) in self.cages.iter().enumerate() {
            if !c.contains(index) {
                continue;
            }
            if value.val() > self.remaining[i] || (c.exclusive && !self.cage_sets[i].contains(uv)) {
                self.illegal = Some((index, value));
            } else {
                self.remaining[i] -= value.val();
                self.empty[i] -= 1;
                self.cage_sets[i].remove(uv);
            }
            break;
        }
        Ok(())
    }

    fn undo(&mut self, index: Index, value: SVal<MIN, MAX>) -> Result<(), Error> {
        if let Some((i, v)) = self.illegal {
            if i != index || v != value {
                return Err(UNDO_MISMATCH);
            } else {
                self.illegal = None;
                return Ok(());
            }
        }
        for (i, c) in self.cages.iter().enumerate() {
            if !c.contains(index) {
                continue;
            }
            self.remaining[i] += value.val();
            self.empty[i] += 1;
            self.cage_sets[i].insert(value.to_uval());
            break;
        }
        Ok(())
    }
}

fn subset_sum(vals: &Vec<u8>, i: usize, remaining: u8, empty: usize) -> bool {
    if i >= vals.len() {
        empty == 0 && remaining == 0
    } else if empty == 0 {
        remaining == 0
    } else if empty == 1 {
        for j in i..vals.len() {
            if vals[j] == remaining {
                return true;
            }
        }
        false
    } else if vals[i] <= remaining && subset_sum(vals, i+1, remaining-vals[i], empty-1) {
        true
    } else {
        subset_sum(vals, i+1, remaining, empty)
    }
}

fn cage_feasible<const MIN: u8, const MAX: u8>(set: &Set<u8>, remaining: u8, empty: usize) -> bool {
    if empty == 0 {
        return remaining == 0;
    } else if set.len() < empty {
        return false;
    }
    let vals = unpack_sval_vals::<MIN, MAX>(set);
    let mut min = 0;
    let mut max = 0;
    for i in 0..empty {
        min += vals[i];
        max += vals[vals.len()-1-i];
    }
    if remaining < min || remaining > max {
        return false;
    }
    // Random choice, but I don't want to bother if the possibilities are huge
    if empty > 5 || (MAX-MIN+1) > 15 {
        return true;
    }
    subset_sum(&vals, 0, remaining, empty)
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize, O: Overlay>
Constraint<u8, SState<N, M, MIN, MAX, O>> for CageChecker<MIN, MAX> {
    fn check(&self, _: &SState<N, M, MIN, MAX, O>, grid: &mut DecisionGrid<u8, SVal<MIN, MAX>>) -> ConstraintResult<u8, SVal<MIN, MAX>> {
        if self.illegal.is_some() {
            return ConstraintResult::Contradiction;
        }
        for (i, c) in self.cages.iter().enumerate() {
            let mut set = if c.exclusive {
                self.cage_sets[i].clone()
            } else {
                full_set::<u8, SVal<MIN, MAX>>()
            };
            if !cage_feasible::<MIN, MAX>(&set, self.remaining[i], self.empty[i]) {
                return ConstraintResult::Contradiction;
            }
            for v in (self.remaining[i]+1)..=MAX {
                set.remove(SVal::<MIN, MAX>::new(v).to_uval());
            }
            for cell in &c.cells {
                let g = &mut grid.get_mut(*cell);
                g.0.intersect_with(&set);
                g.1.add(&self.cage_feature, 1.0);
            }
        }
        ConstraintResult::Ok
    }

    fn explain_contradictions(&self, _: &SState<N, M, MIN, MAX, O>) -> Vec<ConstraintViolationDetail> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::vec;
    use crate::{ranker::LinearRanker, solver::test_util::{assert_contradiction_eq, PuzzleReplay}, sudoku::four_standard_parse};

    #[test]
    fn test_subset_sum() {
        assert!(subset_sum(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 0, 15, 3));
    }

    #[test]
    fn test_cage_checker() {
        // Cage 1 is satisfied (1 + 2 = 3).
        let cage1 = Cage{ cells: vec![[0, 0], [0, 1]], target: 3, exclusive: true };
        // Cage 2 is a failure (3 + 4 != 5).
        let cage2 = Cage{ cells: vec![[1, 2], [1, 3]], target: 5, exclusive: true };
        // Cage 3 is a failure even though it's incomplete (4 > 3).
        let cage3 = Cage{ cells: vec![[2, 0], [2, 1]], target: 3, exclusive: true };
        // Cage 4 has no violations because it's incomplete and not over target.
        let cage4 = Cage{ cells: vec![[3, 2], [3, 3]], target: 5, exclusive: true };

        let puzzle = four_standard_parse(
            "12..\n\
             ..34\n\
             4...\n\
             ..4.\n"
        ).unwrap();

        for (c, expected) in vec![(cage1, false), (cage2, true), (cage3, true), (cage4, false)] {
            let mut puzzle = puzzle.clone();
            let ranker = LinearRanker::default();
            let mut cage_checker = CageChecker::new(vec![c]);
            let result = PuzzleReplay::new(&mut puzzle, &ranker, &mut cage_checker, None).replay().unwrap();
            assert_contradiction_eq(&cage_checker, &puzzle, &result, expected);
        }
    }

    // TODO: Add an e2e cage example.
}