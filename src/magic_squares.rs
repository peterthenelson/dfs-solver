use std::fmt::Debug;
use crate::{constraint::Constraint, core::{empty_set, CertainDecision, ConstraintResult, DecisionGrid, FKWithId, FeatureKey, Index, State, Stateful, UVSet, Value}, sudoku::{Overlay, SState, SVal}};

/// This is _standard, exclusive_ magic square. These are extremely limiting--
/// 5 must go in the middle, odds go on the sides, and evens go in the corners,
/// and (of course, by definition), all the different columns/rows/diagonals
/// add up to the same number, which is 15.
#[derive(Debug, Clone)]
pub struct MagicSquare {
    center: Index,
}

impl MagicSquare {
    pub fn new(center: Index) -> Self { Self { center } }

    pub fn contains(&self, index: Index) -> bool {
        if index == self.center {
            return true;
        }
        let dist = std::cmp::max(
            self.center[0].abs_diff(index[0]),
            self.center[1].abs_diff(index[1]),
        );
        dist <= 1
    }

    pub fn row_0(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r - 1, c - 1], [r - 1, c], [r - 1, c + 1]]
    }

    pub fn row_1(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r, c - 1], [r, c], [r, c + 1]]
    }

    pub fn row_2(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r + 1, c - 1], [r + 1, c], [r + 1, c + 1]]
    }

    pub fn col_0(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r - 1, c - 1], [r, c - 1], [r + 1, c - 1]]
    }

    pub fn col_1(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r - 1, c], [r, c], [r + 1, c]]
    }

    pub fn col_2(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r - 1, c + 1], [r, c + 1], [r + 1, c + 1]]
    }

    pub fn diag_0(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r - 1, c - 1], [r, c], [r + 1, c + 1]]
    }

    pub fn diag_1(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r - 1, c + 1], [r, c], [r + 1, c - 1]]
    }
}

pub const MS_FEATURE: &str = "MAGIC_SQUARE";

pub struct MagicSquareChecker {
    squares: Vec<MagicSquare>,
    evens: UVSet<u8>,
    odds: UVSet<u8>,
    ms_feature: FeatureKey<FKWithId>,
}

impl MagicSquareChecker {
    pub fn new(squares: Vec<MagicSquare>) -> Self {
        let mut evens = empty_set::<u8, SVal<1, 9>>();
        for v in [2, 4, 6, 8] {
            evens.insert(SVal::<1, 9>::new(v).to_uval());
        }
        let mut odds = empty_set::<u8, SVal<1, 9>>();
        for v in [1, 3, 7, 9] {
            odds.insert(SVal::<1, 9>::new(v).to_uval());
        }
        Self {
            squares,
            evens,
            odds,
            ms_feature: FeatureKey::new(MS_FEATURE).unwrap(),
        }
    }
}

impl Debug for MagicSquareChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MagicSquareChecker: {:?}\n", self.squares)
    }
}

impl Stateful<u8, SVal<1, 9>> for MagicSquareChecker {}

fn check_vals<const N: usize, const M: usize, O: Overlay>(
    indices: &[Index; 4],
    values: &UVSet<u8>,
    puzzle: &SState<N, M, 1, 9, O>,
    grid: &mut DecisionGrid<u8, SVal<1, 9>>,
) -> Option<ConstraintResult<u8, SVal<1, 9>>> {
    for i in indices {
        if let Some(v) = puzzle.get(*i) {
            if !values.contains(v.to_uval()) {
                return Some(ConstraintResult::Contradiction)
            }
        } else {
            grid.get_mut(*i).0.intersect_with(values);
        }
    }
    None
}

fn sum_trip<const N: usize, const M: usize, O: Overlay>(
    triple: &[Index; 3], 
    puzzle: &SState<N, M, 1, 9, O>,
) -> (u8, u8, Option<Index>) {
    let mut sum = 0;
    let mut n_empty = 3;
    let mut first_empty = None;
    for i in triple {
        if let Some(v) = puzzle.get(*i) {
            sum += v.val();
            n_empty -= 1;
        } else if first_empty.is_none() {
            first_empty = Some(*i)
        }
    }
    (sum, n_empty, first_empty)
}

fn sum15<const N: usize, const M: usize, O: Overlay>(
    triple: &[Index; 3], 
    puzzle: &SState<N, M, 1, 9, O>,
    grid: &mut DecisionGrid<u8, SVal<1, 9>>,
) -> Option<ConstraintResult<u8, SVal<1, 9>>> {
    let (sum, n_empty, first_empty) = sum_trip(triple, puzzle);
    if n_empty == 0 {
        if sum != 15 {
            Some(ConstraintResult::Contradiction)
        } else {
            None
        }
    } else if n_empty == 1 {
        if sum < 15 {
            let i = first_empty.unwrap();
            let rem = SVal::new(15 - sum);
            if grid.get(i).0.contains(rem.to_uval()) {
                Some(ConstraintResult::Certainty(CertainDecision::new(i, rem)))
            } else {
                Some(ConstraintResult::Contradiction)
            }
        } else {
            Some(ConstraintResult::Contradiction)
        }
    } else {
        None
    }
}

impl <const N: usize, const M: usize, O: Overlay>
Constraint<u8, SState<N, M, 1, 9, O>> for MagicSquareChecker {
    fn check(&self, puzzle: &SState<N, M, 1, 9, O>, grid: &mut DecisionGrid<u8, SVal<1, 9>>) -> ConstraintResult<u8, SVal<1, 9>> {
        for square in &self.squares {
            if let Some(v) = puzzle.get(square.center) {
                if v.val() != 5 {
                    return ConstraintResult::Contradiction;
                }
            } else {
                return ConstraintResult::Certainty(CertainDecision::new(square.center, SVal::new(5)));
            }
            let [ul, _, lr] = square.diag_0();
            let [ll, _, ur] = square.diag_1();
            if let Some(cr) = check_vals(&[ul, lr, ll, ur], &self.evens, puzzle, grid) {
                return cr;
            }
            let [ml, _, mr] = square.row_1();
            let [mu, _, md] = square.col_1();
            if let Some(cr) = check_vals(&[ml, mr, mu, md], &self.odds, puzzle, grid) {
                return cr;
            }
            for triple in [
                square.col_0(),
                square.col_1(),
                square.col_2(),
                square.row_0(),
                square.row_1(),
                square.row_2(),
                square.diag_0(),
                square.diag_1(),
            ] {
                if let Some(cr) = sum15(&triple, puzzle, grid) {
                    return cr;
                }
            }
            for r in 0..3 {
                for c in 0..3 {
                    let i = [square.center[0] - 1 + r, square.center[1] - 1 + c];
                    grid.get_mut(i).1.add(&self.ms_feature, 1.0);
                }
            }
        }
        ConstraintResult::Ok
    }
}

// TODO: Tests