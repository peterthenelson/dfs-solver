use crate::dfs::{Constraint, ConstraintResult, ConstraintViolationDetail, PartialStrategy};
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

pub struct XSumPartialStrategy<const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    xsums: Vec<XSum<MIN, MAX, N, M>>,
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
PartialStrategy<SudokuAction<MIN, MAX>, Sudoku<N, M, MIN, MAX>> for XSumPartialStrategy<MIN, MAX, N, M> {
    fn suggest(&self, puzzle: &Sudoku<N, M, MIN, MAX>) -> Result<Vec<SudokuAction<MIN, MAX>>, crate::dfs::PuzzleError> {
        for xsum in &self.xsums {
            match xsum.length(puzzle) {
                Some(len) => {
                    let mut sum = len;
                    let mut first_empty: Option<(usize, usize)> = None;
                    let mut n_empty = 0;
                    match xsum.direction {
                        XSumDirection::Row => {
                            for i in 1..std::cmp::min(M, len as usize) {
                                match puzzle.grid[xsum.index][i] {
                                    Some(v) => sum += v,
                                    None => {
                                        if first_empty.is_none() {
                                            first_empty = Some((xsum.index, i));
                                        }
                                        n_empty += 1;
                                    }
                                }
                            }
                        },
                        XSumDirection::Col => {
                            for i in 1..std::cmp::min(N, len as usize) {
                                match puzzle.grid[i][xsum.index] {
                                    Some(v) => sum += v,
                                    None => {
                                        if first_empty.is_none() {
                                            first_empty = Some((i, xsum.index));
                                        }
                                        n_empty += 1;
                                    }
                                }
                            }
                        }
                    }
                    if let Some(empty) = first_empty {
                        let target = xsum.target - sum;
                        if n_empty == 1 && MIN <= target && target <= MAX {
                            return Ok(vec![SudokuAction { row: empty.0, col: empty.1, value: target }]);
                        }
                        let mut suggestions = Vec::new();
                        for v in MIN..=std::cmp::min(std::cmp::min(target, MAX), (M + 1) as u8) {
                            suggestions.push(SudokuAction { row: empty.0, col: empty.1, value: v });
                        }
                        return Ok(suggestions);
                    }
                },
                None => {
                    let mut suggestions = Vec::new();
                    match xsum.direction {
                        XSumDirection::Row => {
                            for v in MIN..=std::cmp::min(std::cmp::min(MAX, xsum.target), (M + 1) as u8) {
                                suggestions.push(SudokuAction { row: xsum.index, col: 0, value: v });
                            }
                        },
                        XSumDirection::Col => {
                            for v in MIN..=std::cmp::min(std::cmp::min(MAX, xsum.target), (N + 1) as u8) {
                                suggestions.push(SudokuAction { row: 0, col: xsum.index, value: v });
                            }
                        }
                    }
                    return Ok(suggestions);
                },
            }
        }
        return Ok(vec![]);
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

    #[test]
    fn test_xsum_partial_strategy_no_first_digit() {
        let xsum_strategy = XSumPartialStrategy {
            xsums: vec![
                // Missing first digit, so suggest 1 to 5 (no higher than target).
                XSum { direction: XSumDirection::Col, index: 0, target: 5 },
            ],
        };
        let puzzle = Sudoku::<6, 6, 1, 6>::parse(
            "......\n\
             2.....\n\
             3.....\n\
             ......\n\
             ......\n\
             ......\n"
        ).unwrap();
        let result = xsum_strategy.suggest(&puzzle);
        assert_eq!(result.unwrap(), vec![
            SudokuAction { row: 0, col: 0, value: 1 },
            SudokuAction { row: 0, col: 0, value: 2 },
            SudokuAction { row: 0, col: 0, value: 3 },
            SudokuAction { row: 0, col: 0, value: 4 },
            SudokuAction { row: 0, col: 0, value: 5 },
        ]);
    }

    #[test]
    fn test_xsum_partial_strategy_first_digit_one_empty() {
        let xsum_strategy = XSumPartialStrategy {
            xsums: vec![
                // Has first digit and one remaining empty cell, so suggest the
                // exact answer.
                XSum { direction: XSumDirection::Row, index: 1, target: 7 },
            ],
        };
        let puzzle = Sudoku::<6, 6, 1, 6>::parse(
            "......\n\
             2.....\n\
             3.....\n\
             ......\n\
             ......\n\
             ......\n"
        ).unwrap();
        let result = xsum_strategy.suggest(&puzzle);
        assert_eq!(result.unwrap(), vec![
            SudokuAction { row: 1, col: 1, value: 5 },
        ]);
    }

    #[test]
    fn test_xsum_partial_strategy_first_digit_multiple_remaining() {
        let xsum_strategy = XSumPartialStrategy {
            xsums: vec![
                // Has first digit and multiple empty cells, so suggest 1 to 4
                // (no higher than remaining).
                XSum { direction: XSumDirection::Row, index: 2, target: 7 },
            ],
        };
        let puzzle = Sudoku::<6, 6, 1, 6>::parse(
            "......\n\
             2.....\n\
             3.....\n\
             ......\n\
             ......\n\
             ......\n"
        ).unwrap();
        let result = xsum_strategy.suggest(&puzzle);
        assert_eq!(result.unwrap(), vec![
            SudokuAction { row: 2, col: 1, value: 1 },
            SudokuAction { row: 2, col: 1, value: 2 },
            SudokuAction { row: 2, col: 1, value: 3 },
            SudokuAction { row: 2, col: 1, value: 4 },
        ]);
    }
}