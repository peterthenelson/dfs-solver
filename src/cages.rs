use crate::core::PuzzleError;
use crate::dfs::{Constraint, ConstraintResult, ConstraintViolationDetail, PartialStrategy};
use crate::sudoku::{Sudoku, SudokuAction};

#[derive(Debug, Clone)]
pub struct Cage {
    pub cells: Vec<(usize, usize)>,
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
Constraint<SudokuAction<MIN, MAX>, Sudoku<N, M, MIN, MAX>> for CageChecker {
    fn check(&self, puzzle: &Sudoku<N, M, MIN, MAX>, details: bool) -> ConstraintResult<SudokuAction<MIN, MAX>> {
        let mut violations = Vec::new();
        for cage in self.cages.iter() {
            let mut has_empty = false;
            let mut sum = 0;
            let mut sum_highlight: Vec<SudokuAction<MIN, MAX>> = Vec::new();
            let mut seen = vec![false; (MAX - MIN + 1) as usize];
            let mut seen_highlight: Vec<SudokuAction<MIN, MAX>> = Vec::new();
            for cell in &cage.cells {
                if let Some(value) = puzzle.grid[cell.0][cell.1] {
                    sum += value;
                    if cage.exclusive && seen[(value - MIN) as usize] {
                        if details {
                            seen_highlight.push(SudokuAction{ row: cell.0, col: cell.1, value });
                        } else {
                            return ConstraintResult::Simple("Cage exclusivity violation");
                        }
                    }
                    seen[(value - MIN) as usize] = true;
                    if details {
                        sum_highlight.push(SudokuAction{ row: cell.0, col: cell.1, value });
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
PartialStrategy<SudokuAction<MIN, MAX>, Sudoku<N, M, MIN, MAX>> for CagePartialStrategy {
    fn suggest(&self, puzzle: &Sudoku<N, M, MIN, MAX>) -> Result<Vec<SudokuAction<MIN, MAX>>, PuzzleError> {
        for cage in &self.cages {
            let mut sum = 0;
            let mut first_empty: Option<(usize, usize)> = None;
            let mut n_empty = 0;
            let mut seen = vec![false; (MAX - MIN + 1) as usize];
            for cell in &cage.cells {
                match puzzle.grid[cell.0][cell.1] {
                    Some(value) => {
                        sum += value;
                        if cage.exclusive {
                            seen[(value - MIN) as usize] = true;
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
                    return Ok(vec![SudokuAction {
                        row: empty.0,
                        col: empty.1,
                        value: remaining,
                    }]);
                }
                return Ok((MIN..(std::cmp::min(remaining, MAX) + 1)).filter_map(|value| {
                    return match seen[(value - MIN) as usize] {
                        true => None,
                        false => Some(SudokuAction {
                            row: empty.0,
                            col: empty.1,
                            value,
                        }),
                    };
                }).collect::<Vec<SudokuAction<MIN, MAX>>>());
            }
        }
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use super::*;

    #[test]
    fn test_cage_checker() {
        // Cage 1 is satisfied (1 + 2 = 3).
        let cage1 = Cage{ cells: vec![(0, 0), (0, 1)], target: 3, exclusive: true };
        // Cage 2 is a failure (3 + 4 != 5).
        let cage2 = Cage{ cells: vec![(1, 2), (1, 3)], target: 5, exclusive: true };
        // Cage 3 is a failure even though it's incomplete (4 > 3).
        let cage3 = Cage{ cells: vec![(2, 0), (2, 1)], target: 3, exclusive: true };
        // Cage 4 has no violations because it's incomplete and not over target.
        let cage4 = Cage{ cells: vec![(3, 2), (3, 3)], target: 4, exclusive: true };

        let cage_checker = CageChecker::new(vec![cage1, cage2, cage3, cage4]);
        let puzzle = Sudoku::<4, 4, 1, 4>::parse(
            "12..\n\
             ..34\n\
             4...\n\
             ..4.\n"
        ).unwrap();

        let result = cage_checker.check(&puzzle, true);
        assert_eq!(result, ConstraintResult::Details(vec![
            ConstraintViolationDetail {
                message: "Cage violation: expected sum 5 but got 7".to_string(),
                highlight: Some(vec![
                    SudokuAction { row: 1, col: 2, value: 3 },
                    SudokuAction { row: 1, col: 3, value: 4 },
                ]),
            },
            ConstraintViolationDetail {
                message: "Cage violation: expected sum 3 but got 4".to_string(),
                highlight: Some(vec![
                    SudokuAction { row: 2, col: 0, value: 4 },
                ]),
            },
        ]));
    }

    #[test]
    fn test_cage_partial_strategy_exclusive() {
        let cage_strategy = CagePartialStrategy {
            cages: vec![Cage { cells: vec![(0, 0), (0, 1), (0, 2)], target: 7, exclusive: true }],
        };
        let puzzle = Sudoku::<6, 6, 1, 6>::parse(
            "..4...\n\
             ......\n\
             ......\n\
             ......\n\
             ......\n\
             ......\n"
        ).unwrap();
        let actions = cage_strategy.suggest(&puzzle).unwrap();
        assert_eq!(actions, vec![
            SudokuAction { row: 0, col: 0, value: 1 },
            SudokuAction { row: 0, col: 0, value: 2 },
            SudokuAction { row: 0, col: 0, value: 3 },
        ]);
    }

    #[test]
    fn test_cage_partial_strategy_nonexclusive() {
        let cage_strategy = CagePartialStrategy {
            cages: vec![Cage { cells: vec![(0, 0), (0, 1), (1, 0)], target: 7, exclusive: false }],
        };
        let puzzle = Sudoku::<6, 6, 1, 6>::parse(
            "......\n\
             2.....\n\
             ......\n\
             ......\n\
             ......\n\
             ......\n"
        ).unwrap();
        let actions = cage_strategy.suggest(&puzzle).unwrap();
        assert_eq!(actions, vec![
            SudokuAction { row: 0, col: 0, value: 1 },
            SudokuAction { row: 0, col: 0, value: 2 },
            SudokuAction { row: 0, col: 0, value: 3 },
            SudokuAction { row: 0, col: 0, value: 4 },
            SudokuAction { row: 0, col: 0, value: 5 },
        ]);
    }

    #[test]
    fn test_cage_partial_strategy_last_digit() {
        let cage_strategy = CagePartialStrategy {
            cages: vec![Cage { cells: vec![(0, 0), (0, 1)], target: 6, exclusive: true }],
        };
        let puzzle = Sudoku::<6, 6, 1, 6>::parse(
            "2.....\n\
             ......\n\
             ......\n\
             ......\n\
             ......\n\
             ......\n"
        ).unwrap();
        let actions = cage_strategy.suggest(&puzzle).unwrap();
        assert_eq!(actions, vec![
            SudokuAction { row: 0, col: 1, value: 4 },
        ]);
    }
}