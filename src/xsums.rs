use crate::core::{Error, Index, State};
use crate::constraint::{Constraint, ConstraintResult, ConstraintViolationDetail};
use crate::strategy::{BranchPoint, PartialStrategy};
use crate::sudoku::{SState, SVal};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum XSumDirection {
    // Row, going right (starting from the left)
    RR,
    // Row, going left (starting from the right)
    RL,
    // Column, going down (starting from the top)
    CD,
    // Column, going up (starting from the bottom)
    CU,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct XSum<const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    pub target: u8,
    pub index: usize,
    pub direction: XSumDirection,
}

pub struct XSumIter<'a, const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    xsum: &'a XSum<MIN, MAX, N, M>,
    r: isize,
    c: isize,
    // Inclusive limit for the iterator
    lim: isize,
}

impl <'a, const MIN: u8, const MAX: u8, const N: usize, const M: usize> XSumIter<'a, MIN, MAX, N, M> {
    pub fn new(xsum: &'a XSum<MIN, MAX, N, M>, r: usize, c: usize, lim: usize) -> Self {
        XSumIter { xsum, r: r as isize, c: c as isize, lim: lim as isize }
    }
}

impl <'a, const MIN: u8, const MAX: u8, const N: usize, const M: usize> Iterator for XSumIter<'a, MIN, MAX, N, M> {
    type Item = Index;

    fn next(&mut self) -> Option<Self::Item> {
        if self.r < 0 || self.c < 0 {
            return None;
        }
        let ret = Some([self.r as usize, self.c as usize]);
        match self.xsum.direction {
            XSumDirection::RR => {
                if self.c > self.lim {
                    return None;
                } else {
                    self.c += 1;
                    return ret;
                }
            },
            XSumDirection::RL => {
                if self.c < self.lim {
                    return None;
                } else {
                    self.c -= 1;
                    return ret;
                }
            },
            XSumDirection::CD => {
                if self.r > self.lim {
                    return None;
                } else {
                    self.r += 1;
                    return ret;
                }
            },
            XSumDirection::CU => {
                if self.r < self.lim {
                    return None;
                } else {
                    self.r -= 1;
                    return ret;
                }
            },
        }
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize> XSum<MIN, MAX, N, M> {
    pub fn length(&self, puzzle: &SState<N, M, MIN, MAX>) -> Option<(Index, SVal<MIN, MAX>)> {
        match self.direction {
            XSumDirection::RR => if let Some(v) = puzzle.get([self.index, 0]) {
                Some(([self.index, 0], v))
            } else {
                None
            },
            XSumDirection::RL => if let Some(v) = puzzle.get([self.index, M - 1]) {
                Some(([self.index, M - 1], v))
            } else {
                None
            },
            XSumDirection::CD => if let Some(v) = puzzle.get([0, self.index]) {
                Some(([0, self.index], v))
            } else {
                None
            },
            XSumDirection::CU => if let Some(v) = puzzle.get([N - 1, self.index]) {
                Some(([N - 1, self.index], v))
            } else {
                None
            },
        }
    }

    pub fn length_index(&self) -> Index {
        match self.direction {
            XSumDirection::RR => [self.index, 0],
            XSumDirection::RL => [self.index, M - 1],
            XSumDirection::CD => [0, self.index],
            XSumDirection::CU => [N - 1, self.index],
        }
    }

    /// Returns an iterator over the range of indices (OTHER than the length
    /// digit) that are part of the XSum. E.g., for an XSum for row index 2
    /// going right with a length digit of 3, the iterator will return (2, 1)
    /// and (2, 2).
    pub fn xrange(&self, length: u8) -> XSumIter<MIN, MAX, N, M> {
        match self.direction {
            XSumDirection::RR => XSumIter::new(self, self.index, 1, std::cmp::min(M, length as usize) - 1),
            XSumDirection::RL => XSumIter::new(self, self.index, M - 2, std::cmp::max(0, (M as i16) - (length as i16)) as usize),
            XSumDirection::CD => XSumIter::new(self, 1, self.index, std::cmp::min(N, length as usize) - 1),
            XSumDirection::CU => XSumIter::new(self, N - 2, self.index, std::cmp::max(0, (N as i16) - (length as i16)) as usize),
        }
    }
}

pub struct XSumChecker<const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    pub xsums: Vec<XSum<MIN, MAX, N, M>>,
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize> XSumChecker<MIN, MAX, N, M> {
    pub fn new(xsums: Vec<XSum<MIN, MAX, N, M>>) -> Self {
        XSumChecker { xsums }
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Constraint<u8, SState<N, M, MIN, MAX>> for XSumChecker<MIN, MAX, N, M> {
    fn check(&self, puzzle: &SState<N, M, MIN, MAX>, details: bool) -> ConstraintResult {
        let mut violations = Vec::new();
        for xsum in self.xsums.iter() {
            if let Some((len_index, len)) = xsum.length(puzzle) {
                let mut sum = len.val();
                let mut sum_highlight = Vec::new();
                let mut has_empty = false;
                if details {
                    sum_highlight.push(len_index);
                }
                for [r, c] in xsum.xrange(len.val()) {
                    if let Some(v) = puzzle.get([r, c]) {
                        sum += v.val();
                        if details {
                            sum_highlight.push([r, c]);
                        }
                    } else {
                        has_empty = true;
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
    pub xsums: Vec<XSum<MIN, MAX, N, M>>,
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
PartialStrategy<u8, SState<N, M, MIN, MAX>> for XSumPartialStrategy<MIN, MAX, N, M> {
    fn suggest_partial(&self, puzzle: &SState<N, M, MIN, MAX>) -> Result<BranchPoint<u8, SState<N, M, MIN, MAX>, std::vec::IntoIter<SVal<MIN, MAX>>>, Error> {
        for xsum in &self.xsums {
            match xsum.length(puzzle) {
                Some((_, len)) => {
                    let mut sum = len.val();
                    let mut first_empty: Option<Index> = None;
                    let mut n_empty = 0;
                    for [r, c] in xsum.xrange(len.val()) {
                        match puzzle.get([r, c]) {
                            Some(v) => sum += v.val(),
                            None => {
                                if first_empty.is_none() {
                                    first_empty = Some([r, c]);
                                }
                                n_empty += 1;
                            }
                        }
                    }
                    if let Some(empty) = first_empty {
                        let target = xsum.target - sum;
                        if n_empty == 1 && MIN <= target && target <= MAX {
                            return Ok(BranchPoint::unique(empty, SVal::new(target)));
                        }
                        let mut suggestions = Vec::new();
                        for v in MIN..=std::cmp::min(std::cmp::min(target, MAX), (M + 1) as u8) {
                            suggestions.push(SVal::new(v));
                        }
                        return Ok(BranchPoint::new(empty, suggestions.into_iter()));
                    }
                },
                None => {
                    let mut suggestions = Vec::new();
                    let max = match xsum.direction {
                        XSumDirection::RR | XSumDirection::RL => std::cmp::min(
                            xsum.target,
                            std::cmp::min(M as u8, MAX)),
                        XSumDirection::CD | XSumDirection::CU => std::cmp::min(
                            xsum.target,
                            std::cmp::min(N as u8, MAX)),
                    };
                    for v in MIN..=max {
                        suggestions.push(SVal::new(v));
                    }
                    return Ok(BranchPoint::new(xsum.length_index(), suggestions.into_iter()));
                },
            }
        }
        return Ok(BranchPoint::empty());
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use super::*;

    #[test]
    fn test_xsum_length() {
        let x1 = XSum { direction: XSumDirection::RR, index: 0, target: 5 };
        let x2 = XSum { direction: XSumDirection::RL, index: 0, target: 5 };
        let x3 = XSum { direction: XSumDirection::CD, index: 0, target: 5 };
        let x4 = XSum { direction: XSumDirection::CU, index: 0, target: 5 };
        let puzzle1 = SState::<4, 4, 1, 4>::parse(
            "2..3\n\
             ....\n\
             ....\n\
             4...\n"
        ).unwrap();
        assert_eq!(x1.length(&puzzle1).unwrap(), ([0, 0], SVal::new(2)));
        assert_eq!(x2.length(&puzzle1).unwrap(), ([0, 3], SVal::new(3)));
        assert_eq!(x3.length(&puzzle1).unwrap(), ([0, 0], SVal::new(2)));
        assert_eq!(x4.length(&puzzle1).unwrap(), ([3, 0], SVal::new(4)));
        let puzzle2 = SState::<4, 4, 1, 4>::parse(
            "....\n\
             .21.\n\
             .43.\n\
             ....\n"
        ).unwrap();
        assert!(x1.length(&puzzle2).is_none());
        assert!(x2.length(&puzzle2).is_none());
        assert!(x3.length(&puzzle2).is_none());
        assert!(x4.length(&puzzle2).is_none());
    }

    #[test]
    fn test_xsum_xrange() {
        let x1: XSum<1, 4, 4, 4> = XSum { direction: XSumDirection::RR, index: 0, target: 5 };
        let x2: XSum<1, 4, 4, 4> = XSum { direction: XSumDirection::RL, index: 0, target: 5 };
        let x3: XSum<1, 4, 4, 4> = XSum { direction: XSumDirection::CD, index: 0, target: 5 };
        let x4: XSum<1, 4, 4, 4> = XSum { direction: XSumDirection::CU, index: 0, target: 5 };
        // Imagine this puzzle:
        // 2..3
        // ....
        // ....
        // 4...
        assert_eq!(x1.xrange(2).collect::<Vec<_>>(), vec![[0, 1]]);
        assert_eq!(x2.xrange(3).collect::<Vec<_>>(), vec![[0, 2], [0, 1]]);
        assert_eq!(x3.xrange(2).collect::<Vec<_>>(), vec![[1, 0]]);
        assert_eq!(x4.xrange(4).collect::<Vec<_>>(), vec![[2, 0], [1, 0], [0, 0]]);
    }

    #[test]
    fn test_xsum_checker_rr_cd() {
        // XSum 1 is satisfied (2 + 1 = 3).
        let x1 = XSum{ direction: XSumDirection::RR, index: 0, target: 3 };
        // XSum 2 is a failure (3 + 4 + 2 != 10).
        let x2 = XSum{ direction: XSumDirection::CD, index: 2, target: 10 };
        // XSum 3 is a failure even though it's incomplete (4 + 3 + 2 > 5).
        let x3 = XSum{ direction: XSumDirection::RR, index: 2, target: 5 };
        // XSum 4 has no violations because it's incomplete and not over target.
        let x4 = XSum{ direction: XSumDirection::CD, index: 3, target: 7 };
        // XSum 5 has no violations because it doesn't even have a first digit yet.
        let x5 = XSum{ direction: XSumDirection::RR, index: 3, target: 1 };

        let xsum_checker = XSumChecker::new(vec![x1, x2, x3, x4, x5]);
        let puzzle = SState::<4, 4, 1, 4>::parse(
            "2134\n\
             ..4.\n\
             432.\n\
             ....\n"
        ).unwrap();

        let result = xsum_checker.check(&puzzle, true);
        assert_eq!(result, ConstraintResult::Details(vec![
            ConstraintViolationDetail {
                message: "X sum violation: expected sum 10 but got 9".to_string(),
                highlight: Some(vec![[0, 2], [1, 2], [2, 2]]),
            },
            ConstraintViolationDetail {
                message: "X sum violation: expected sum 5 but got 9".to_string(),
                highlight: Some(vec![[2, 0], [2, 1], [2, 2]]),
            },
        ]));
    }

    #[test]
    fn test_xsum_checker_rl_cu() {
        // XSum 1 is satisfied (2 + 1 = 3).
        let x1 = XSum{ direction: XSumDirection::RL, index: 3, target: 3 };
        // XSum 2 is a failure (3 + 4 + 2 != 10).
        let x2 = XSum{ direction: XSumDirection::CU, index: 1, target: 10 };
        // XSum 3 is a failure even though it's incomplete (4 + 3 + 2 > 5).
        let x3 = XSum{ direction: XSumDirection::RL, index: 1, target: 5 };
        // XSum 4 has no violations because it's incomplete and not over target.
        let x4 = XSum{ direction: XSumDirection::CU, index: 0, target: 7 };
        // XSum 5 has no violations because it doesn't even have a first digit yet.
        let x5 = XSum{ direction: XSumDirection::RL, index: 0, target: 1 };

        let xsum_checker = XSumChecker::new(vec![x1, x2, x3, x4, x5]);
        let puzzle = SState::<4, 4, 1, 4>::parse(
            "....\n\
             .234\n\
             .4..\n\
             4312\n"
        ).unwrap();

        let result = xsum_checker.check(&puzzle, true);
        assert_eq!(result, ConstraintResult::Details(vec![
            ConstraintViolationDetail {
                message: "X sum violation: expected sum 10 but got 9".to_string(),
                highlight: Some(vec![[3, 1], [2, 1], [1, 1]]),
            },
            ConstraintViolationDetail {
                message: "X sum violation: expected sum 5 but got 9".to_string(),
                highlight: Some(vec![[1, 3], [1, 2], [1, 1]]),
            },
        ]));
    }

    #[test]
    fn test_xsum_partial_strategy_no_first_digit() {
        // All are missing the first digit, so suggestions range from 1 to the
        // target (or MAX).
        let xsum_strategy1 = XSumPartialStrategy {
            xsums: vec![
                XSum { direction: XSumDirection::RR, index: 1, target: 3 },
            ],
        };
        let xsum_strategy2 = XSumPartialStrategy {
            xsums: vec![
                XSum { direction: XSumDirection::RL, index: 1, target: 4 },
            ],
        };
        let xsum_strategy3 = XSumPartialStrategy {
            xsums: vec![
                XSum { direction: XSumDirection::CD, index: 1, target: 5 },
            ],
        };
        let xsum_strategy4 = XSumPartialStrategy {
            xsums: vec![
                XSum { direction: XSumDirection::CU, index: 1, target: 7 },
            ],
        };
        let puzzle = SState::<6, 6, 1, 6>::parse(
            "......\n\
             .2....\n\
             .3....\n\
             ......\n\
             ......\n\
             ......\n"
        ).unwrap();
        let d = xsum_strategy1.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [1, 0]);
        assert_eq!(d.into_vec(), vec![
            SVal::new(1),
            SVal::new(2),
            SVal::new(3),
        ]);
        let d = xsum_strategy2.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [1, 5]);
        assert_eq!(d.into_vec(), vec![
            SVal::new(1),
            SVal::new(2),
            SVal::new(3),
            SVal::new(4),
        ]);
        let d = xsum_strategy3.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [0, 1]);
        assert_eq!(d.into_vec(), vec![
            SVal::new(1),
            SVal::new(2),
            SVal::new(3),
            SVal::new(4),
            SVal::new(5),
        ]);
        let d = xsum_strategy4.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [5, 1]);
        assert_eq!(d.into_vec(), vec![
            SVal::new(1),
            SVal::new(2),
            SVal::new(3),
            SVal::new(4),
            SVal::new(5),
            SVal::new(6),
        ]);
    }

    #[test]
    fn test_xsum_partial_strategy_first_digit_one_empty() {
        // Each of these has a first digit and one empty cell, so suggest the
        // exact answer.
        let xsum_strategy1 = XSumPartialStrategy {
            xsums: vec![
                XSum { direction: XSumDirection::RR, index: 1, target: 3 },
            ],
        };
        let xsum_strategy2 = XSumPartialStrategy {
            xsums: vec![
                XSum { direction: XSumDirection::CD, index: 1, target: 3 },
            ],
        };
        let xsum_strategy3 = XSumPartialStrategy {
            xsums: vec![
                XSum { direction: XSumDirection::RL, index: 2, target: 9 },
            ],
        };
        let xsum_strategy4 = XSumPartialStrategy {
            xsums: vec![
                XSum { direction: XSumDirection::CU, index: 4, target: 10 },
            ],
        };
        let puzzle = SState::<6, 6, 1, 6>::parse(
            ".2....\n\
             2.....\n\
             ....23\n\
             ......\n\
             ....3.\n\
             ....4.\n"
        ).unwrap();
        let d = xsum_strategy1.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [1, 1]);
        assert_eq!(d.into_vec(), vec![SVal::new(1)]);
        let d = xsum_strategy2.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [1, 1]);
        assert_eq!(d.into_vec(), vec![SVal::new(1)]);
        let d = xsum_strategy3.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [2, 3]);
        assert_eq!(d.into_vec(), vec![SVal::new(4)]);
        let d = xsum_strategy4.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [3, 4]);
        assert_eq!(d.into_vec(), vec![SVal::new(1)]);
    }

    #[test]
    fn test_xsum_partial_strategy_first_digit_multiple_remaining() {
        // Each of these has a first digit and multiple empty cells, so suggest
        // 1 to the target-current (or MAX) for the first empty cell.
        let xsum_strategy1 = XSumPartialStrategy {
            xsums: vec![
                XSum { direction: XSumDirection::RR, index: 1, target: 5 },
            ],
        };
        let xsum_strategy2 = XSumPartialStrategy {
            xsums: vec![
                XSum { direction: XSumDirection::CD, index: 1, target: 6 },
            ],
        };
        let xsum_strategy3 = XSumPartialStrategy {
            xsums: vec![
                XSum { direction: XSumDirection::RL, index: 1, target: 7 },
            ],
        };
        let xsum_strategy4 = XSumPartialStrategy {
            xsums: vec![
                XSum { direction: XSumDirection::CU, index: 4, target: 12 },
            ],
        };
        let puzzle = SState::<6, 6, 1, 6>::parse(
            ".4....\n\
             4....3\n\
             ......\n\
             ......\n\
             ......\n\
             ....3.\n"
        ).unwrap();
        let d = xsum_strategy1.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [1, 1]);
        assert_eq!(d.into_vec(), vec![SVal::new(1)]);
        let d = xsum_strategy2.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [1, 1]);
        assert_eq!(d.into_vec(), vec![SVal::new(1), SVal::new(2)]);
        let d = xsum_strategy3.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [1, 4]);
        assert_eq!(d.into_vec(), vec![SVal::new(1), SVal::new(2), SVal::new(3), SVal::new(4)]);
        let d = xsum_strategy4.suggest_partial(&puzzle).unwrap();
        assert_eq!(d.index, [4, 4]);
        assert_eq!(d.into_vec(), vec![
            SVal::new(1),
            SVal::new(2),
            SVal::new(3),
            SVal::new(4),
            SVal::new(5),
            SVal::new(6),
        ]);
    }
}