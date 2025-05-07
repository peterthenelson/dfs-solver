use crate::{dfs::{Constraint, ConstraintResult, ConstraintViolationDetail}, sudoku::{Sudoku, SudokuAction}};

pub struct Cage {
    pub cells: Vec<(usize, usize)>,
    pub target: u8,
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
}