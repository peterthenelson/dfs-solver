use crate::{dfs::{Constraint, ConstraintResult, ConstraintViolationDetail, PartialStrategy}, sudoku::{Sudoku, SudokuAction}};

pub struct Cage {
    pub cells: Vec<(usize, usize)>,
    pub target: u8,
    // TODO: Add option for exclusivity
}

impl Cage {
    pub fn new(cells: Vec<(usize, usize)>, target: u8) -> Self {
        Cage { cells, target }
    }

    pub fn get_cells(&self) -> &Vec<(usize, usize)> {
        &self.cells
    }

    pub fn get_target(&self) -> u8 {
        self.target
    }
}

pub struct CageChecker {
    cages: Vec<Cage>,
}

impl CageChecker {
    pub fn new(cages: Vec<Cage>) -> Self {
        CageChecker { cages }
    }

    pub fn get_cages(&self) -> &Vec<Cage> {
        &self.cages
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Constraint<SudokuAction<MIN, MAX>, Sudoku<N, M, MIN, MAX>> for CageChecker {
    fn check(&self, puzzle: &Sudoku<N, M, MIN, MAX>, details: bool) -> ConstraintResult<SudokuAction<MIN, MAX>> {
        let mut violations = Vec::new();
        for cage in self.cages.iter() {
            let mut sum = 0;
            let mut has_empty = false;
            let mut highlight = Vec::new();
            for cell in &cage.cells {
                if let Some(value) = puzzle.grid[cell.0][cell.1] {
                    sum += value;
                    if details {
                        highlight.push(SudokuAction{ row: cell.0, col: cell.1, value });
                    }
                } else {
                    has_empty = true;
                }
            }
            if (has_empty && sum > cage.get_target()) || (!has_empty && sum != cage.get_target()) {
                if details {
                    violations.push(ConstraintViolationDetail {
                        message: format!("Cage violation: expected sum {} but got {}", cage.get_target(), sum),
                        highlight: Some(highlight),
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
    cages: Vec<Cage>,
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
PartialStrategy<SudokuAction<MIN, MAX>, Sudoku<N, M, MIN, MAX>> for CagePartialStrategy {
    // TODO: Give exact value if it's the last cell in the cage.
    // TODO: Update the strategy for cages that are exclusive.
    fn suggest(&self, puzzle: &Sudoku<N, M, MIN, MAX>) -> Result<Vec<SudokuAction<MIN, MAX>>, crate::dfs::PuzzleError> {
        for cage in &self.cages {
            let mut sum = 0;
            let mut first_empty: Option<(usize, usize)> = None;
            for cell in &cage.cells {
                match puzzle.grid[cell.0][cell.1] {
                    Some(value) => sum += value,
                    None => {
                        if first_empty.is_none() {
                            first_empty = Some(cell.clone());
                        }
                    }
                }
            }
            if let Some(empty) = first_empty {
                let target = cage.get_target();
                let remaining = target - sum;
                return Ok((MIN..(std::cmp::min(remaining, MAX) + 1)).map(|value| SudokuAction {
                    row: empty.0,
                    col: empty.1,
                    value,
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
    fn test_cage_creation() {
        let cells = vec![(0, 0), (0, 1), (1, 0)];
        let target = 6;
        let cage = Cage::new(cells.clone(), target);
        assert_eq!(cage.get_cells(), &cells);
        assert_eq!(cage.get_target(), target);
    }

    #[test]
    fn test_cage_checker() {
        // Cage 1 is satisfied (1 + 2 = 3).
        let cage1 = Cage::new(vec![(0, 0), (0, 1)], 3);
        // Cage 2 is a failure (3 + 4 != 5).
        let cage2 = Cage::new(vec![(1, 2), (1, 3)], 5);
        // Cage 3 is a failure even though it's incomplete (4 > 3).
        let cage3 = Cage::new(vec![(2, 0), (2, 1)], 3);
        // Cage 4 has no violations because it's incomplete and not over target.
        let cage4 = Cage::new(vec![(3, 2), (3, 3)], 5);

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
    fn test_cage_partial_strategy() {
        let cage_strategy = CagePartialStrategy {
            cages: vec![Cage::new(vec![(0, 0), (0, 1)], 4)],
        };
        let puzzle = Sudoku::<4, 4, 1, 4>::parse(
            "2...\n\
             ....\n\
             ....\n\
             ....\n"
        ).unwrap();
        let actions = cage_strategy.suggest(&puzzle).unwrap();
        assert_eq!(actions, vec![
            SudokuAction { row: 0, col: 1, value: 1 },
            SudokuAction { row: 0, col: 1, value: 2 },
        ]);
    }
}