use std::collections::HashSet;
use std::fmt::Debug;
use crate::core::{empty_set, full_set, DecisionGrid, Error, Index, Set, State, Stateful, Value};
use crate::constraint::{Constraint, ConstraintResult, ConstraintViolationDetail};
use crate::strategy::{BranchPoint, PartialStrategy};
use crate::sudoku::{unpack_sval_vals, SState, SVal};

#[derive(Debug, Clone)]
pub struct Cage {
    pub cells: Vec<Index>,
    pub target: u8,
    pub exclusive: bool,
}

impl Cage {
    pub fn contains(&self, index: Index) -> bool {
        self.cells.contains(&index)
    }
}

pub const ILLEGAL_ACTION_CAGE: Error = Error::new_const("A cage violation already exists; can't apply further actions.");
pub const UNDO_MISMATCH: Error = Error::new_const("Undo value mismatch");

pub struct CageChecker <const MIN: u8, const MAX: u8> {
    cages: Vec<Cage>,
    remaining: Vec<u8>,
    cage_sets: Vec<Set<u8>>,
    illegal: Option<(Index, SVal<MIN, MAX>)>,
}

impl <const MIN: u8, const MAX: u8> CageChecker<MIN, MAX> {
    // Note: Cages must not overlap
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
        let cage_sets = vec![full_set::<u8, SVal<MIN, MAX>>(); cages.len()];
        CageChecker { cages, remaining, cage_sets, illegal: None }
    }
}

impl <const MIN: u8, const MAX: u8> Debug for CageChecker<MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((i, v)) = self.illegal {
            write!(f, "Illegal move: {:?}; {:?}\n", i, v)?;
        }
        for (i, c) in self.cages.iter().enumerate() {
            write!(f, " Cage[{:?}]\n", c.cells)?;
            write!(f, " - Remaining to target: {}\n", self.remaining[i])?;
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
            self.cage_sets[i].insert(value.to_uval());
            break;
        }
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Constraint<u8, SState<N, M, MIN, MAX>> for CageChecker<MIN, MAX> {
    fn check(&self, _: &SState<N, M, MIN, MAX>, force_grid: bool) -> ConstraintResult<u8, SVal<MIN, MAX>> {
        if self.illegal.is_some() {
            if force_grid {
                return ConstraintResult::grid(DecisionGrid::new(N, M));
            }
            return ConstraintResult::Contradiction;
        }
        let mut grid = DecisionGrid::full(N, M);
        for (i, c) in self.cages.iter().enumerate() {
            let mut set = if c.exclusive {
                self.cage_sets[i].clone()
            } else {
                full_set::<u8, SVal<MIN, MAX>>()
            };
            for v in (self.remaining[i]+1)..=MAX {
                set.remove(SVal::<MIN, MAX>::new(v).to_uval());
            }
            for cell in &c.cells {
                grid.get_mut(*cell).0 = set.clone();
            }
        }
        ConstraintResult::grid(grid)
    }

    fn explain_contradictions(&self, _: &SState<N, M, MIN, MAX>) -> Vec<ConstraintViolationDetail> {
        todo!()
    }
}

pub struct CagePartialStrategy {
    pub cages: Vec<Cage>,
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
PartialStrategy<u8, SState<N, M, MIN, MAX>> for CagePartialStrategy {
    fn suggest_partial(&self, puzzle: &SState<N, M, MIN, MAX>) -> Result<BranchPoint<u8, SState<N, M, MIN, MAX>, std::vec::IntoIter<SVal<MIN, MAX>>>, Error> {
        for cage in &self.cages {
            let mut sum = 0;
            let mut first_empty: Option<Index> = None;
            let mut n_empty = 0;
            let mut seen = empty_set::<u8, SVal<MIN, MAX>>();
            for cell in &cage.cells {
                match puzzle.get(*cell) {
                    Some(value) => {
                        sum += value.val();
                        if cage.exclusive {
                            seen.insert(value.to_uval());
                        }
                    },
                    None => {
                        if first_empty.is_none() {
                            first_empty = Some(cell.clone());
                        }
                        n_empty += 1;
                    }
                }
            }
            if let Some(empty) = first_empty {
                let target = cage.target;
                let remaining = target - sum;
                if n_empty == 1 && MIN <= remaining && remaining <= MAX {
                    return Ok(BranchPoint::unique(empty, SVal::new(remaining)));
                }
                let actions = (MIN..(std::cmp::min(remaining, MAX) + 1)).filter_map(|value| {
                    let value = SVal::<MIN, MAX>::new(value);
                    return if seen.contains(value.to_uval()) {
                        None
                    } else {
                        Some(value)
                    };
                }).collect::<Vec<SVal<MIN, MAX>>>();
                return Ok(BranchPoint::new(empty, actions.into_iter()));
            }
        }
        Ok(BranchPoint::empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;
    use crate::solver::test_util::{assert_contradiction_eq, replay_puzzle};

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

        let puzzle = SState::<4, 4, 1, 4>::parse(
            "12..\n\
             ..34\n\
             4...\n\
             ..4.\n"
        ).unwrap();

        for (c, expected) in vec![(cage1, false), (cage2, true), (cage3, true), (cage4, false)] {
            let mut cage_checker = CageChecker::new(vec![c]);
            let result = replay_puzzle(&mut cage_checker, &puzzle, false);
            assert_contradiction_eq(&cage_checker, &puzzle, &result, expected);
        }
    }

    #[test]
    fn test_cage_partial_strategy_exclusive() {
        let cage_strategy = CagePartialStrategy {
            cages: vec![Cage { cells: vec![[0, 0], [0, 1], [0, 2]], target: 7, exclusive: true }],
        };
        let puzzle = SState::<6, 6, 1, 6>::parse(
            "..4...\n\
             ......\n\
             ......\n\
             ......\n\
             ......\n\
             ......\n"
        ).unwrap();
        let d = cage_strategy.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [0, 0]);
        assert_eq!(d.into_vec(), vec![SVal::new(1), SVal::new(2), SVal::new(3)]);
    }

    #[test]
    fn test_cage_partial_strategy_nonexclusive() {
        let cage_strategy = CagePartialStrategy {
            cages: vec![Cage { cells: vec![[0, 0], [0, 1], [1, 0]], target: 7, exclusive: false }],
        };
        let puzzle = SState::<6, 6, 1, 6>::parse(
            "......\n\
             2.....\n\
             ......\n\
             ......\n\
             ......\n\
             ......\n"
        ).unwrap();
        let d = cage_strategy.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [0, 0]);
        assert_eq!(d.into_vec(), (1..=5).map(SVal::new).collect::<Vec<SVal<1, 6>>>());
    }

    #[test]
    fn test_cage_partial_strategy_last_digit() {
        let cage_strategy = CagePartialStrategy {
            cages: vec![Cage { cells: vec![[0, 0], [0, 1]], target: 6, exclusive: true }],
        };
        let puzzle = SState::<6, 6, 1, 6>::parse(
            "2.....\n\
             ......\n\
             ......\n\
             ......\n\
             ......\n\
             ......\n"
        ).unwrap();
        let d = cage_strategy.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [0, 1]);
        assert_eq!(d.into_vec(), vec![SVal::new(4)]);
    }

    // TODO: Add an e2e cage example.
}