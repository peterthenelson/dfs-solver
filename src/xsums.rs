use crate::dfs::{Constraint, ConstraintResult, ConstraintViolationDetail};
use crate::sudoku::{Sudoku, SudokuAction};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum XSumDirection { Row, Col }

pub struct XSum<const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    pub target: u8,
    pub index: usize,
    pub direction: XSumDirection,
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize> XSum<MIN, MAX, N, M> {
    pub fn length(&self, puzzle: &Sudoku<N, M, MIN, MAX>) -> Option<u8> {
        match self.direction {
            XSumDirection::Row => puzzle.grid[self.index][0],
            XSumDirection::Col => puzzle.grid[0][self.index],
        }
    }

    pub fn bound(&self, puzzle: &Sudoku<N, M, MIN, MAX>) -> usize {
        match self.direction {
            XSumDirection::Row => match puzzle.grid[self.index][0] {
                Some(v) => std::cmp::min(M, v as usize),
                None => M,
            },
            XSumDirection::Col => match puzzle.grid[0][self.index] {
                Some(v) => std::cmp::min(N, v as usize),
                None => N,
            }
        }
    }
}

pub struct XSumChecker<const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    xsums: Vec<XSum<MIN, MAX, N, M>>,
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize> XSumChecker<MIN, MAX, N, M> {
    pub fn new(xsums: Vec<XSum<MIN, MAX, N, M>>) -> Self {
        XSumChecker { xsums }
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Constraint<SudokuAction<MIN, MAX>, Sudoku<N, M, MIN, MAX>> for XSumChecker<MIN, MAX, N, M> {
    fn check(&self, puzzle: &Sudoku<N, M, MIN, MAX>, details: bool) -> ConstraintResult<SudokuAction<MIN, MAX>> {
        let mut violations = Vec::new();
        for xsum in self.xsums.iter() {
            if let Some(len) = xsum.length(puzzle) {
                let mut sum = len;
                let mut sum_highlight = Vec::new();
                let mut has_empty = false;
                match xsum.direction {
                    XSumDirection::Row => {
                        if details {
                            sum_highlight.push(SudokuAction { row: xsum.index, col: 0, value: len });
                        }
                        for i in 1..std::cmp::min(M, len as usize) {
                            if let Some(v) = puzzle.grid[xsum.index][i] {
                                sum += v;
                                if details {
                                    sum_highlight.push(SudokuAction { row: xsum.index, col: i, value: v });
                                }
                            } else {
                                has_empty = true;
                            }
                        }
                    },
                    XSumDirection::Col => {
                        if details {
                            sum_highlight.push(SudokuAction { row: 0, col: xsum.index, value: len });
                        }
                        for i in 1..std::cmp::min(N, len as usize) {
                            if let Some(v) = puzzle.grid[i][xsum.index] {
                                sum += v;
                                if details {
                                    sum_highlight.push(SudokuAction { row: i, col: xsum.index, value: v });
                                }
                            } else {
                                has_empty = true;
                            }
                        }
                    }
                }
                if sum > xsum.target || (!has_empty && sum != xsum.target) {
                    if details {
                        violations.push(ConstraintViolationDetail {
                            message: format!("X sum violation: expected sum {} but got {}", xsum.target, sum),
                            highlight: Some(sum_highlight),
                        })
                    } else {
                        return ConstraintResult::Simple("X sum violation");
                    }
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
    fn test_xsum_checker() {
        // XSum 1 is satisfied (2 + 1 = 3).
        let x1 = XSum{ direction: XSumDirection::Row, index: 0, target: 3 };
        // XSum 2 is a failure (3 + 4 + 2 != 10).
        let x2 = XSum{ direction: XSumDirection::Col, index: 2, target: 10 };
        // XSum 3 is a failure even though it's incomplete (4 + 3 + 2 > 5).
        let x3 = XSum{ direction: XSumDirection::Row, index: 2, target: 5 };
        // XSum 4 has no violations because it's incomplete and not over target.
        let x4 = XSum{ direction: XSumDirection::Col, index: 3, target: 7 };
        // XSum 5 has no violations because it doesn't even have a first digit yet.
        let x5 = XSum{ direction: XSumDirection::Row, index: 3, target: 1 };

        let xsum_checker = XSumChecker::new(vec![x1, x2, x3, x4, x5]);
        let puzzle = Sudoku::<4, 4, 1, 4>::parse(
            "2134\n\
             ..4.\n\
             432.\n\
             ....\n"
        ).unwrap();

        let result = xsum_checker.check(&puzzle, true);
        assert_eq!(result, ConstraintResult::Details(vec![
            ConstraintViolationDetail {
                message: "X sum violation: expected sum 10 but got 9".to_string(),
                highlight: Some(vec![
                    SudokuAction { row: 0, col: 2, value: 3 },
                    SudokuAction { row: 1, col: 2, value: 4 },
                    SudokuAction { row: 2, col: 2, value: 2 },
                ]),
            },
            ConstraintViolationDetail {
                message: "X sum violation: expected sum 5 but got 9".to_string(),
                highlight: Some(vec![
                    SudokuAction { row: 2, col: 0, value: 4 },
                    SudokuAction { row: 2, col: 1, value: 3 },
                    SudokuAction { row: 2, col: 2, value: 2 },
                ]),
            },
        ]));
    }
}