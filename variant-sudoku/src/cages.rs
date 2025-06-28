use std::collections::HashSet;
use std::fmt::Debug;
use crate::core::{Attribution, ConstraintResult, Error, Feature, Index, Key, Overlay, State, Stateful, VBitSet, VSet, VSetMut};
use crate::constraint::Constraint;
use crate::index_util::{check_orthogonally_connected};
use crate::ranker::RankingInfo;
use crate::sudoku::{unpack_stdval_vals, StdVal};

#[derive(Debug, Clone)]
pub struct Cage {
    pub target: Option<u8>,
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

    fn check(cage: Cage) -> Cage {
        if !cage.exclusive && cage.target.is_none() {
            panic!("Cage with neither target nor exclusivity: {:?}\nThis is \
                    probably a mistake; purely decorative cages should not \
                    be included when building your constraints.", cage);
        }
        check_orthogonally_connected(&cage.cells).unwrap();
        cage
    }

    pub fn sum(&self, target: u8, cells: Vec<Index>) -> Cage {
        let exclusive = self.0 || self.1.all_mutually_visible(&cells);
        Self::check(Cage { target: Some(target), cells, exclusive })
    }

    pub fn nosum(&self, cells: Vec<Index>) -> Cage {
        let exclusive = self.0 || self.1.all_mutually_visible(&cells);
        Self::check(Cage { target: None, cells, exclusive })
    }

    pub fn across(&self, target: u8, left: Index, length: usize) -> Cage {
        let cells = (0..length)
            .map(|i| [left[0], left[1]+i])
            .collect();
        self.sum(target, cells)
    }

    pub fn down(&self, target: u8, top: Index, length: usize) -> Cage {
        let cells = (0..length)
            .map(|i| [top[0]+i, top[1]])
            .collect();
        self.sum(target, cells)
    }

    pub fn rect(&self, target: u8, top_left: Index, bottom_right: Index) -> Cage {
        let mut cells = vec![];
        for r in top_left[0]..=bottom_right[0] {
            for c in top_left[1]..=bottom_right[1] {
                cells.push([r, c]);
            }
        }
        self.sum(target, cells)
    }

    pub fn v(&self, cell1: Index, cell2: Index) -> Cage {
        // This should be trivially true, but let's defer to the visibility
        // checker rather than assume it.
        let exclusive = self.0 || self.1.mutually_visible(cell1, cell2);
        Self::check(Cage { target: Some(5), cells: vec![cell1, cell2], exclusive })
    }

    pub fn x(&self, cell1: Index, cell2: Index) -> Cage {
        // This should be trivially true, but let's defer to the visibility
        // checker rather than assume it.
        let exclusive = self.0 || self.1.mutually_visible(cell1, cell2);
        Self::check(Cage { target: Some(10), cells: vec![cell1, cell2], exclusive })
    }
}

pub const CAGE_ILLEGAL_ACTION: Error = Error::new_const("A cage violation already exists; can't apply further actions.");
pub const CAGE_UNDO_MISMATCH: Error = Error::new_const("Undo value mismatch");
pub const CAGE_FEATURE: &str = "CAGE";
pub const CAGE_DUPE_ATTRIBUTION: &str = "CAGE_DUPE";
pub const CAGE_OVER_ATTRIBUTION: &str = "CAGE_SUM_OVER";
pub const CAGE_INFEASIBLE_ATTRIBUTION: &str = "CAGE_SUM_INFEASIBLE";

pub struct CageChecker<const MIN: u8, const MAX: u8> {
    cages: Vec<Cage>,
    remaining: Vec<Option<u8>>,
    empty: Vec<usize>,
    cage_sets: Vec<VBitSet<StdVal<MIN, MAX>>>,
    cage_feature: Key<Feature>,
    cage_dupe_attr: Key<Attribution>,
    cage_over_attr: Key<Attribution>,
    cage_if_attr: Key<Attribution>,
    illegal: Option<(Index, StdVal<MIN, MAX>, Key<Attribution>)>,
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
        let cage_sets = vec![VBitSet::<StdVal<MIN, MAX>>::full(); cages.len()];
        CageChecker {
            cages, remaining, empty, cage_sets, illegal: None,
            cage_feature: Key::register(CAGE_FEATURE),
            cage_dupe_attr: Key::register(CAGE_DUPE_ATTRIBUTION),
            cage_over_attr: Key::register(CAGE_OVER_ATTRIBUTION),
            cage_if_attr: Key::register(CAGE_INFEASIBLE_ATTRIBUTION),
        }
    }
}

impl <const MIN: u8, const MAX: u8> Debug for CageChecker<MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((i, v, a)) = &self.illegal {
            write!(f, "Illegal move: {:?}={:?} ({})\n", i, v, a.name())?;
        }
        for (i, c) in self.cages.iter().enumerate() {
            write!(f, " Cage{:?}\n", c.cells)?;
            write!(f, " - {:?} remaining to target; {} cells empty\n", self.remaining[i], self.empty[i])?;
            if c.exclusive {
                let vals = unpack_stdval_vals::<MIN, MAX, _>(&self.cage_sets[i]);
                write!(f, " - Unused vals: {:?}\n", vals)?;
            }
        }
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8> Stateful<StdVal<MIN, MAX>> for CageChecker<MIN, MAX> {
    fn reset(&mut self) {
        self.remaining = self.cages.iter().map(|c| c.target).collect();
        self.empty = self.cages.iter().map(|c| c.cells.len()).collect();
        self.cage_sets = vec![VBitSet::<StdVal<MIN, MAX>>::full(); self.cages.len()];
        self.illegal = None;
    }

    fn apply(&mut self, index: Index, value: StdVal<MIN, MAX>) -> Result<(), Error> {
        // In theory we could be allow multiple illegal moves and just
        // invalidate and recalculate the grid or something, but it seems hard.
        if self.illegal.is_some() {
            return Err(CAGE_ILLEGAL_ACTION);
        }
        for (i, c) in self.cages.iter().enumerate() {
            if !c.contains(index) {
                continue;
            }
            if let Some(r) = self.remaining[i] {
                if value.val() > r {
                    self.illegal = Some((index, value, self.cage_over_attr));
                    break;
                } else if c.exclusive && !self.cage_sets[i].contains(&value) {
                    self.illegal = Some((index, value, self.cage_dupe_attr));
                    break;
                }
            }
            self.remaining[i].as_mut().map(|i| *i -= value.val());
            self.empty[i] -= 1;
            self.cage_sets[i].remove(&value);
            break;
        }
        Ok(())
    }

    fn undo(&mut self, index: Index, value: StdVal<MIN, MAX>) -> Result<(), Error> {
        if let Some((i, v, _)) = self.illegal {
            if i != index || v != value {
                return Err(CAGE_UNDO_MISMATCH);
            } else {
                self.illegal = None;
                return Ok(());
            }
        }
        for (i, c) in self.cages.iter().enumerate() {
            if !c.contains(index) {
                continue;
            }
            self.remaining[i].as_mut().map(|i| *i += value.val());
            self.empty[i] += 1;
            self.cage_sets[i].insert(&value);
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

fn cage_feasible<const MIN: u8, const MAX: u8>(set: &VBitSet<StdVal<MIN, MAX>>, remaining: u8, empty: usize) -> bool {
    if empty == 0 {
        return remaining == 0;
    } else if set.len() < empty {
        return false;
    }
    let vals = unpack_stdval_vals::<MIN, MAX, _>(set);
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

impl <const MIN: u8, const MAX: u8, O: Overlay>
Constraint<StdVal<MIN, MAX>, O> for CageChecker<MIN, MAX> {
    fn check(&self, puzzle: &State<StdVal<MIN, MAX>, O>, ranking: &mut RankingInfo<StdVal<MIN, MAX>>) -> ConstraintResult<StdVal<MIN, MAX>> {
        let grid = ranking.cells_mut();
        if let Some((_, _, a)) = &self.illegal {
            return ConstraintResult::Contradiction(*a);
        }
        for (i, c) in self.cages.iter().enumerate() {
            let mut set = if c.exclusive {
                self.cage_sets[i].clone()
            } else {
                VBitSet::<StdVal<MIN, MAX>>::full()
            };
            if let Some(r) = self.remaining[i] {
                if !cage_feasible::<MIN, MAX>(&set, r, self.empty[i]) {
                    return ConstraintResult::Contradiction(self.cage_if_attr);
                }
                for v in (r+1)..=MAX {
                    set.remove(&StdVal::<MIN, MAX>::new(v));
                }
            }
            for cell in &c.cells {
                let g = &mut grid.get_mut(*cell);
                if puzzle.get(*cell).is_none() {
                    g.0.intersect_with(&set);
                }
                g.1.add(&self.cage_feature, 1.0);
            }
        }
        ConstraintResult::Ok
    }

    fn debug_at(&self, _: &State<StdVal<MIN, MAX>, O>, index: Index) -> Option<String> {
        let header = "CageChecker:\n";
        let mut lines = vec![];
        if let Some((i, v, a)) = &self.illegal {
            if *i == index {
                lines.push(format!("  Illegal move: {:?}={:?} ({})", i, v, a.name()));
            }
        }
        for (i, c) in self.cages.iter().enumerate() {
            if !c.contains(index) {
                continue;
            }
            lines.push(format!("  Cage{:?}", c.cells));
            if let Some(r) = self.remaining[i] {
                lines.push(format!("  - {} remaining to target", r));
            }
            lines.push(format!("  - {} cells empty", self.empty[i]));
            if c.exclusive {
                let vals = unpack_stdval_vals::<MIN, MAX, _>(&self.cage_sets[i]);
                lines.push(format!("  - Unused vals: {:?}", vals));
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
    use super::*;
    use std::vec;
    use crate::{constraint::{test_util::*, MultiConstraint}, ranker::StdRanker, solver::{test_util::PuzzleReplay, FindFirstSolution, PuzzleSetter}, sudoku::{four_standard_overlay, four_standard_parse, nine_standard_overlay, FourStd, FourStdOverlay, FourStdVal, StdChecker}};

    #[test]
    fn test_subset_sum() {
        assert!(subset_sum(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 0, 15, 3));
    }

    #[test]
    #[should_panic]
    fn test_cage_build_orthogonally_connected() {
        let vis = nine_standard_overlay();
        let cb = CageBuilder::new(true, &vis);
        cb.sum(10, vec![[0, 0], [0, 1], [0, 3], [0, 4]]);
    }

    #[test]
    fn test_cage_checker() {
        // Cage 1 is satisfied (1 + 2 = 3).
        let cage1 = Cage{ cells: vec![[0, 0], [0, 1]], target: Some(3), exclusive: true };
        // Cage 2 is a failure (3 + 4 != 5).
        let cage2 = Cage{ cells: vec![[1, 2], [1, 3]], target: Some(5), exclusive: true };
        // Cage 3 is a failure even though it's incomplete (4 > 3).
        let cage3 = Cage{ cells: vec![[2, 0], [2, 1]], target: Some(3), exclusive: true };
        // Cage 4 has no violations because it's incomplete and not over target.
        let cage4 = Cage{ cells: vec![[3, 2], [3, 3]], target: Some(5), exclusive: true };
        // Cage 5 is a failure even though it's incomplete (not possible to reach target)
        let cage5 = Cage{ cells: vec![[0, 0], [1, 0]], target: Some(6), exclusive: true };
        // Cage 6 is a failure because of a dupe.
        let cage6 = Cage{ cells: vec![[0, 1], [0, 2]], target: Some(5), exclusive: true };

        let puzzle = four_standard_parse(
            "122.\n\
             ..34\n\
             4...\n\
             ..4.\n"
        ).unwrap();

        for (c, expected) in vec![
            (cage1, None),
            (cage2, Some("CAGE_SUM_OVER")),
            (cage3, Some("CAGE_SUM_OVER")),
            (cage4, None),
            (cage5, Some("CAGE_SUM_INFEASIBLE")),
            (cage6, Some("CAGE_DUPE")),
        ] {
            let mut puzzle = puzzle.clone();
            let ranker = StdRanker::default();
            let mut cage_checker = CageChecker::new(vec![c]);
            let result = PuzzleReplay::new(&mut puzzle, &ranker, &mut cage_checker, None).replay().unwrap();
            if let Some(attr) = expected {
                assert_contradiction(result, attr);
            } else {
                assert_no_contradiction(result);
            }
        }
    }

    // https://appynation.helpshift.com/hc/en/13-puzzle-page/faq/290-killer-sudoku/
    struct E2ECage;
    impl PuzzleSetter for E2ECage {
        type Value = FourStdVal;
        type Overlay = FourStdOverlay;
        type Ranker = StdRanker;
        type Constraint = MultiConstraint<Self::Value, Self::Overlay>;
        fn setup() -> (FourStd, Self::Ranker, Self::Constraint) {
            Self::setup_with_givens(FourStd::new(four_standard_overlay()))
        }
        fn setup_with_givens(given: FourStd) -> (FourStd, Self::Ranker, Self::Constraint) {
            let cb = CageBuilder::new(true, given.overlay());
            let cages = vec![
                cb.across(8, [0, 0], 3),
                cb.down(5, [0, 3], 2),
                cb.down(6, [1, 0], 2),
                cb.across(1, [1, 1], 1),
                cb.sum(8, vec![[1, 2], [2, 2], [2, 3]]),
                cb.sum(6, vec![[3, 0], [3, 1], [2, 1]]),
                cb.across(6, [3, 2], 2),
            ];
            let constraint = MultiConstraint::new(vec_box::vec_box![
                StdChecker::new(&given),
                CageChecker::new(cages),
            ]);
            (given, StdRanker::default(), constraint)
        }
    }

    #[test]
    fn test_e2e_cage_example() -> Result<(), Error> {
        let (mut puzzle, ranker, mut constraint) = E2ECage::setup();
        let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, None);
        let maybe_solution = finder.solve()?;
        assert!(maybe_solution.is_some());
        let expected: &str = "3412\n\
                              2143\n\
                              4231\n\
                              1324\n";
        assert_eq!(format!("{:?}", maybe_solution.unwrap().state()), expected);
        Ok(())
    }
}