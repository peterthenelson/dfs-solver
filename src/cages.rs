use crate::core::{empty_set, Error, Index, State, Value};
use crate::constraint::{Constraint, ConstraintResult, ConstraintViolationDetail};
use crate::strategy::{DecisionPoint, PartialStrategy};
use crate::sudoku::{SState, SVal};

#[derive(Debug, Clone)]
pub struct Cage {
    pub cells: Vec<Index>,
    pub target: u8,
    pub exclusive: bool,
}

pub struct CageChecker {
    cages: Vec<Cage>,
}

impl CageChecker {
    pub fn new(cages: Vec<Cage>) -> Self {
        CageChecker { cages }
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Constraint<u8, SState<N, M, MIN, MAX>> for CageChecker {
    fn check(&self, puzzle: &SState<N, M, MIN, MAX>, details: bool) -> ConstraintResult {
        let mut violations = Vec::new();
        for cage in self.cages.iter() {
            let mut has_empty = false;
            let mut sum = 0;
            let mut sum_highlight: Vec<Index> = Vec::new();
            let mut seen = empty_set::<u8, SVal<MIN, MAX>>();
            let mut seen_highlight: Vec<Index> = Vec::new();
            for cell in &cage.cells {
                if let Some(value) = puzzle.get(*cell) {
                    sum += value.val();
                    if cage.exclusive && seen.contains(value.to_uval()) {
                        if details {
                            seen_highlight.push(cell.clone());
                        } else {
                            return ConstraintResult::Simple("Cage exclusivity violation");
                        }
                    }
                    seen.insert(value.to_uval());
                    if details {
                        sum_highlight.push(cell.clone());
                    }
                } else {
                    has_empty = true;
                }
            }
            if (has_empty && sum > cage.target) || (!has_empty && sum != cage.target) {
                if details {
                    violations.push(ConstraintViolationDetail {
                        message: format!("Cage violation: expected sum {} but got {}", cage.target, sum),
                        highlight: Some(sum_highlight),
                    });
                } else {
                    return ConstraintResult::Simple("Cage sum violation");
                }
            }
        }
        return if violations.is_empty() {
            ConstraintResult::NoViolation
        } else {
            ConstraintResult::Details(violations)
        }
    }
}

pub struct CagePartialStrategy {
    pub cages: Vec<Cage>,
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
PartialStrategy<u8, SState<N, M, MIN, MAX>> for CagePartialStrategy {
    fn suggest_partial(&self, puzzle: &SState<N, M, MIN, MAX>) -> Result<DecisionPoint<u8, SState<N, M, MIN, MAX>, std::vec::IntoIter<SVal<MIN, MAX>>>, Error> {
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
                    return Ok(DecisionPoint::unique(empty, SVal::new(remaining)));
                }
                let actions = (MIN..(std::cmp::min(remaining, MAX) + 1)).filter_map(|value| {
                    let value = SVal::<MIN, MAX>::new(value);
                    return if seen.contains(value.to_uval()) {
                        None
                    } else {
                        Some(value)
                    };
                }).collect::<Vec<SVal<MIN, MAX>>>();
                return Ok(DecisionPoint::new(empty, actions.into_iter()));
            }
        }
        Ok(DecisionPoint::empty())
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use super::*;

    #[test]
    fn test_cage_checker() {
        // Cage 1 is satisfied (1 + 2 = 3).
        let cage1 = Cage{ cells: vec![[0, 0], [0, 1]], target: 3, exclusive: true };
        // Cage 2 is a failure (3 + 4 != 5).
        let cage2 = Cage{ cells: vec![[1, 2], [1, 3]], target: 5, exclusive: true };
        // Cage 3 is a failure even though it's incomplete (4 > 3).
        let cage3 = Cage{ cells: vec![[2, 0], [2, 1]], target: 3, exclusive: true };
        // Cage 4 has no violations because it's incomplete and not over target.
        let cage4 = Cage{ cells: vec![[3, 2], [3, 3]], target: 4, exclusive: true };

        let cage_checker = CageChecker::new(vec![cage1, cage2, cage3, cage4]);
        let puzzle = SState::<4, 4, 1, 4>::parse(
            "12..\n\
             ..34\n\
             4...\n\
             ..4.\n"
        ).unwrap();

        let result = cage_checker.check(&puzzle, true);
        assert_eq!(result, ConstraintResult::Details(vec![
            ConstraintViolationDetail {
                message: "Cage violation: expected sum 5 but got 7".to_string(),
                highlight: Some(vec![[1, 2], [1, 3]]),
            },
            ConstraintViolationDetail {
                message: "Cage violation: expected sum 3 but got 4".to_string(),
                highlight: Some(vec![[2, 0]]),
            },
        ]));
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
}