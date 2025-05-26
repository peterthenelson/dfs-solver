use std::fmt::Debug;
use crate::core::{empty_set, DecisionGrid, Error, Index, State, Stateful, Value};
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

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize> Debug for XSum<MIN, MAX, N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "XSum({}, {:?}, {})", self.target, self.index, match self.direction {
            XSumDirection::RR => "RR",
            XSumDirection::RL => "RL",
            XSumDirection::CD => "CD",
            XSumDirection::CU => "CU",
        })
    }
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
    pub fn contains_with_len(&self, length: u8, index: Index) -> bool {
        match self.direction {
            XSumDirection::RR => {
                index[0] == self.index && index[1] < length as usize
            },
            XSumDirection::RL => {
                index[0] == self.index && (
                    index[1] as i16 > (M as i16 - 1 - length as i16)
                )
            },
            XSumDirection::CD => {
                index[1] == self.index && index[0] < length as usize
            },
            XSumDirection::CU => {
                index[1] == self.index && (
                    index[0] as i16 > (N as i16 - 1 - length as i16)
                )
            },
        }
    }

    pub fn contains(&self, puzzle: &SState<N, M, MIN, MAX>, index: Index) -> bool {
        let length_index = self.length_index();
        if index == length_index {
            true
        } else if let Some(v) = puzzle.get(length_index) {
            self.contains_with_len(v.val(), index)
        } else {
            false
        }
    }

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

// TODO: Tables for min/max sums for various lengths and vice versa

pub struct XSumChecker<const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    xsums: Vec<XSum<MIN, MAX, N, M>>,
    // Remaining to the target
    xsums_remaining: Vec<i16>,
    // Remaining empty NON-LENGTH cells, supposing length is already known
    xsums_empty: Vec<Option<i16>>,
    // To calculate remaining and empty when a length cell becomes known.
    grid: Vec<Option<SVal<MIN, MAX>>>,
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize> XSumChecker<MIN, MAX, N, M> {
    pub fn new(xsums: Vec<XSum<MIN, MAX, N, M>>) -> Self {
        let xsums_remaining = xsums.iter().map(|x| x.target as i16).collect();
        let xsums_empty = vec![None; xsums.len()];
        let grid = vec![None; N * M];
        XSumChecker { xsums, xsums_remaining, xsums_empty, grid }
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Debug for XSumChecker<MIN, MAX, N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, x) in self.xsums.iter().enumerate() {
            write!(f, " {:?}\n", x)?;
            write!(f, " - Remaining to target: {}\n", self.xsums_remaining[i])?;
            if let Some(empty) = self.xsums_empty[i] {
                write!(f, " - Empty cells remaining: {}", empty)?;
            } else {
                write!(f, " - Length unknown")?;
            }
        }
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Stateful<u8, SVal<MIN, MAX>> for XSumChecker<MIN, MAX, N, M> {
    fn reset(&mut self) {
        self.xsums_remaining = self.xsums.iter().map(|x| x.target as i16).collect();
        self.xsums_empty = vec![None; self.xsums.len()];
        self.grid.fill(None);
    }

    fn apply(&mut self, index: Index, value: SVal<MIN, MAX>) -> Result<(), Error> {
        self.grid[index[0] * M + index[1]] = Some(value);
        for (i, xsum) in self.xsums.iter().enumerate() {
            let len_index = xsum.length_index();
            if index == len_index {
                let mut remaining = xsum.target as i16;
                let mut empty = value.val() as i16 - 1;
                for i2 in xsum.xrange(value.val()) {
                    if let Some(v) = self.grid[i2[0] * M + i2[1]] {
                        remaining -= v.val() as i16;
                        empty -= 1;
                    }
                }
                self.xsums_remaining[i] = remaining;
                self.xsums_empty[i] = Some(empty);
            } else if let Some(len) = self.grid[len_index[0] * M + len_index[1]] {
                if xsum.contains_with_len(len.val(), index) {
                    self.xsums_remaining[i] -= value.val() as i16;
                    self.xsums_empty[i] = self.xsums_empty[i].map(|e| e - 1);
                }
            }
        }
        Ok(())
    }

    fn undo(&mut self, index: Index, value: SVal<MIN, MAX>) -> Result<(), Error> {
        self.grid[index[0] * M + index[1]] = None;
        for (i, xsum) in self.xsums.iter().enumerate() {
            let len_index = xsum.length_index();
            if index == len_index {
                // Reset the state for xsum[i]
                self.xsums_remaining[i] = xsum.target as i16;
                self.xsums_empty[i] = None;
            } else if let Some(len) = self.grid[len_index[0] * M + len_index[1]] {
                if xsum.contains_with_len(len.val(), index) {
                    self.xsums_remaining[i] += value.val() as i16;
                    self.xsums_empty[i] = self.xsums_empty[i].map(|e| e + 1);
                }
            }
        }
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Constraint<u8, SState<N, M, MIN, MAX>> for XSumChecker<MIN, MAX, N, M> {
    fn check(&self, puzzle: &SState<N, M, MIN, MAX>, force_grid: bool) -> ConstraintResult<u8, SVal<MIN, MAX>> {
        let mut grid = DecisionGrid::full(N, M);
        for (i, xsum) in self.xsums.iter().enumerate() {
            if let Some(e) = self.xsums_empty[i] {
                let r = self.xsums_remaining[i];
                if r < 0 || (e == 0 && r != 0) {
                    if !force_grid {
                        return ConstraintResult::Contradiction;
                    }
                    grid = DecisionGrid::new(N, M);
                    break;
                }
                // TODO: Can constrain this more
                let mut set = empty_set::<u8, SVal<MIN, MAX>>();
                (MIN..=(r as u8)).for_each(|v| {
                    set.insert(SVal::<MIN, MAX>::new(v).to_uval())
                });
                let len = xsum.length(puzzle).unwrap().1;
                for i2 in xsum.xrange(len.val()) {
                    grid.get_mut(i2).0 = set.clone();
                }
            }
            // TODO: else we can constrain length based on sums
        }
        ConstraintResult::grid(grid)
    }

    fn explain_contradictions(&self, _: &SState<N, M, MIN, MAX>) -> Vec<ConstraintViolationDetail> {
        todo!()
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
    use super::*;
    use std::vec;
    use crate::constraint::test_util::{assert_contradiction_eq, replay_puzzle};

    #[test]
    fn test_xsum_contains() {
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
        // x1 contains two cells -- the length digit and the next
        assert!(x1.contains(&puzzle1, [0, 0]));
        assert!(x1.contains(&puzzle1, [0, 1]));
        assert!(!x1.contains(&puzzle1, [0, 2]));
        // x2 contains three cells -- the length digit and the prev 2
        assert!(x2.contains(&puzzle1, [0, 3]));
        assert!(x2.contains(&puzzle1, [0, 1]));
        assert!(!x2.contains(&puzzle1, [0, 0]));
        // x3 contains two cells -- the length digit and the next
        assert!(x3.contains(&puzzle1, [0, 0]));
        assert!(x3.contains(&puzzle1, [1, 0]));
        assert!(!x3.contains(&puzzle1, [2, 0]));
        // x3 contains four cells -- the length digit and the rest
        assert!(x4.contains(&puzzle1, [3, 0]));
        assert!(x4.contains(&puzzle1, [0, 0]));
        let puzzle2 = SState::<4, 4, 1, 4>::parse(
            "....\n\
             .21.\n\
             .43.\n\
             ....\n"
        ).unwrap();
        // These all contain their length cell, but since it's empty, they don't
        // contain any other cells.
        assert!(x1.contains(&puzzle2, [0, 0]));
        assert!(!x1.contains(&puzzle2, [0, 1]));
        assert!(x2.contains(&puzzle2, [0, 3]));
        assert!(!x2.contains(&puzzle2, [0, 2]));
        assert!(x3.contains(&puzzle2, [0, 0]));
        assert!(!x3.contains(&puzzle2, [1, 0]));
        assert!(x4.contains(&puzzle2, [3, 0]));
        assert!(!x4.contains(&puzzle2, [2, 0]));
    }

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

        let puzzle = SState::<4, 4, 1, 4>::parse(
            "2134\n\
             ..4.\n\
             432.\n\
             ....\n"
        ).unwrap();

        for (x, expected) in vec![(x1, false), (x2, true), (x3, true), (x4, false), (x5, false)] {
            let mut xsum_checker = XSumChecker::new(vec![x]);
            let result = replay_puzzle(&mut xsum_checker, &puzzle, false);
            assert_contradiction_eq(&xsum_checker, &puzzle, &result, expected);
        }
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

        let puzzle = SState::<4, 4, 1, 4>::parse(
            "....\n\
             .234\n\
             .4..\n\
             4312\n"
        ).unwrap();

        for (x, expected) in vec![(x1, false), (x2, true), (x3, true), (x4, false), (x5, false)] {
            let mut xsum_checker = XSumChecker::new(vec![x]);
            let result = replay_puzzle(&mut xsum_checker, &puzzle, false);
            assert_contradiction_eq(&xsum_checker, &puzzle, &result, expected);
        }
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