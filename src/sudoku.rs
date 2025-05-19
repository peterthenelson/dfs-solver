use std::fmt::Display;

use crate::core::{empty_set, to_value, Error, Index, State, UVGrid, UVal, UValUnwrapped, UValWrapped, Value};
use crate::constraint::{Constraint, ConstraintConjunction, ConstraintResult, ConstraintViolationDetail};
use crate::strategy::{DecisionPoint, Strategy};

/// Standard Sudoku value, ranging from a minimum to a maximum value (inclusive).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SVal<const MIN: u8, const MAX: u8>(u8);

impl <const MIN: u8, const MAX: u8> SVal<MIN, MAX> {
    pub fn new(value: u8) -> Self {
        assert!(value >= MIN && value <= MAX, "Value out of bounds");
        SVal(value)
    }

    pub fn val(self) -> u8 {
        self.0
    }
}

impl <const MIN: u8, const MAX: u8> Value<u8> for SVal<MIN, MAX> {
    fn parse(s: &str) -> Result<Self, Error> {
        let value = s.parse::<u8>().map_err(|v| Error::new(format!("Invalid value: {}", v).to_string()))?;
        if value < MIN || value > MAX {
            return Err(Error::new(format!("Value out of bounds: {} ({}-{})", value, MIN, MAX)));
        }
        Ok(SVal(value))
    }

    fn cardinality() -> usize {
        (MAX - MIN + 1) as usize
    }

    fn from_uval(u: UVal<u8, UValUnwrapped>) -> Self {
        SVal(u.value() + MIN)
    }

    fn to_uval(self) -> UVal<u8, UValWrapped> {
        UVal::new(self.0 - MIN)
    }
}

/// Standard rectangular Sudoku grid.
#[derive(Debug, Clone)]
pub struct SState<const N: usize, const M: usize, const MIN: u8, const MAX: u8> {
    grid: UVGrid<u8>,
} 

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> SState<N, M, MIN, MAX> {
    pub fn new() -> Self {
        Self { grid: UVGrid::new(N, M) }
    }

    pub fn parse(s: &str) -> Result<Self, Error> {
        let mut grid = UVGrid::new(N, M);
        let lines: Vec<&str> = s.lines().collect();
        if lines.len() != N {
            return Err(Error::new("Invalid number of rows".to_string()));
        }
        for i in 0..N {
            let line = lines[i].trim();
            if line.len() != M {
                return Err(Error::new("Invalid number of columns".to_string()));
            }
            for j in 0..M {
                let c = line.chars().nth(j).unwrap();
                if c == '.' {
                    // Already None
                } else {
                    let s = c.to_string();
                    let v = SVal::<MIN, MAX>::parse(s.as_str())?;
                    grid.set([i, j], Some(v.to_uval()));
                }
            }
        }
        Ok(Self { grid })
    }

    pub fn serialize(&self) -> String {
        let mut result = String::new();
        for r in 0..N {
            for c in 0..M {
                if let Some(v) = self.grid.get([r, c]) {
                    result.push_str(to_value::<u8, SVal<MIN, MAX>>(v).val().to_string().as_str());
                } else {
                    result.push('.');
                }
            }
            result.push('\n');
        }
        result
    }
}

pub fn nine_standard_parse(s: &str) -> Result<SState<9, 9, 1, 9>, Error> {
    SState::<9, 9, 1, 9>::parse(s)
}

pub fn eight_standard_parse(s: &str) -> Result<SState<8, 8, 1, 8>, Error> {
    SState::<8, 8, 1, 8>::parse(s)
}

pub fn six_standard_parse(s: &str) -> Result<SState<6, 6, 1, 6>, Error> {
    SState::<6, 6, 1, 6>::parse(s)
}

pub fn four_standard_parse(s: &str) -> Result<SState<4, 4, 1, 4>, Error> {
    SState::<4, 4, 1, 4>::parse(s)
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Display
for SState<N, M, MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        for r in 0..N {
            for c in 0..M {
                if let Some(v) = self.grid.get([r, c]) {
                    write!(f, "{}", to_value::<u8, SVal::<MIN, MAX>>(v).val())?;
                } else {
                    write!(f, ".")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub const OUT_OF_BOUNDS_ERROR: Error = Error::new_const("Out of bounds");
pub const ALREADY_FILLED_ERROR: Error = Error::new_const("Cell already filled");
pub const NO_SUCH_ACTION_ERROR: Error = Error::new_const("No such action to undo");
pub const UNDO_MISMATCH: Error = Error::new_const("Undo value mismatch");

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> State<u8> for SState<N, M, MIN, MAX> {
    type Value = SVal<MIN, MAX>;
    const ROWS: usize = N;
    const COLS: usize = M;

    fn reset(&mut self) {
        self.grid = UVGrid::new(N, M);
    }

    fn get(&self, index: Index) -> Option<Self::Value> {
        if index[0] >= N || index[1] >= M {
            return None;
        }
        self.grid.get(index).map(to_value)
    }

    fn apply(&mut self, index: Index, value: Self::Value) -> Result<(), Error> {
        if index[0] >= N || index[1] >= M {
            return Err(OUT_OF_BOUNDS_ERROR);
        }
        if self.grid.get(index).is_some() {
            return Err(ALREADY_FILLED_ERROR);
        }
        self.grid.set(index, Some(value.to_uval()));
        Ok(())
    }

    fn undo(&mut self, index: Index, value: Self::Value) -> Result<(), Error> {
        if index[0] >= N || index[1] >= M {
            return Err(OUT_OF_BOUNDS_ERROR);
        }
        match self.grid.get(index) {
            None => return Err(NO_SUCH_ACTION_ERROR),
            Some(v) => {
                if v != value.to_uval() {
                    return Err(UNDO_MISMATCH);
                }
            }
        }
        self.grid.set(index, None);
        Ok(())
    }
}

pub struct RowColChecker {}
impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Constraint<u8, SState<N, M, MIN, MAX>> for RowColChecker {
    fn check(&self, puzzle: &SState<N, M, MIN, MAX>, details: bool) -> ConstraintResult {
        let mut violations = vec![];
        for r in 0..N {
            let mut seen = empty_set::<u8, SVal<MIN, MAX>>();
            for c in 0..M {
                if let Some(value) = puzzle.get([r, c]) {
                    if seen.contains(value.to_uval()) {
                        if details {
                            violations.push(ConstraintViolationDetail {
                                message: format!("Duplicate value {} in row {}", value.val(), r),
                                highlight: Some(vec![[r, c]]),
                            })
                        } else {
                            return ConstraintResult::Simple("Duplicate value in row");
                        }
                    }
                    seen.insert(value.to_uval());
                }
            }
        }
        for c in 0..M {
            let mut seen = empty_set::<u8, SVal<MIN, MAX>>();
            for r in 0..N {
                if let Some(value) = puzzle.get([r, c]) {
                    if seen.contains(value.to_uval()) {
                        if details {
                            violations.push(ConstraintViolationDetail {
                                message: format!("Duplicate value {} in col {}", value.val(), c),
                                highlight: Some(vec![[r, c]]),
                            })
                        } else {
                            return ConstraintResult::Simple("Duplicate value in col");
                        }
                    }
                    seen.insert(value.to_uval());
                }
            }
        }
        if violations.len() > 0 {
            return ConstraintResult::Details(violations)
        } else {
            return ConstraintResult::NoViolation;
        }
    }
}

pub struct BoxChecker {
    pub br: usize,
    pub bc: usize,
    pub bh: usize,
    pub bw: usize,
}
impl BoxChecker {
    pub fn new(br: usize, bc: usize, bh: usize, bw: usize) -> Self {
        Self { br, bc, bh, bw }
    }
}
impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Constraint<u8, SState<N, M, MIN, MAX>> for BoxChecker {
    fn check(&self, puzzle: &SState<N, M, MIN, MAX>, details: bool) -> ConstraintResult {
        _ = assert!(N == self.br * self.bh && M == self.bc * self.bw, "Sudoku dimensions do not match box dimensions");
        let mut violations = vec![];
        for box_row in 0..self.br {
            for box_col in 0..self.bc {
                let mut seen = empty_set::<u8, SVal<MIN, MAX>>();
                for r in 0..self.bh {
                    for c in 0..self.bw {
                        let row = box_row * self.bh + r;
                        let col = box_col * self.bw + c;
                        if let Some(value) = puzzle.get([row, col]) {
                            if seen.contains(value.to_uval()) {
                                if details {
                                    violations.push(ConstraintViolationDetail {
                                        message: format!("Duplicate value {} in box ({}, {})", value.val(), box_row, box_col),
                                        highlight: Some(vec![[row, col]]),
                                    })
                                } else {
                                    return ConstraintResult::Simple("Duplicate value in box");
                                }
                            }
                            seen.insert(value.to_uval());
                        }
                    }
                }
            }
        }
        if violations.len() > 0 {
            ConstraintResult::Details(violations)
        } else {
            ConstraintResult::NoViolation
        }
    }
}

pub struct NineBoxChecker(BoxChecker);
impl NineBoxChecker {
    pub fn new() -> Self {
        NineBoxChecker(BoxChecker::new(3, 3, 3, 3))
    }
}
impl Constraint<u8, SState<9, 9, 1, 9>> for NineBoxChecker {
    fn check(&self, puzzle: &SState<9, 9, 1, 9>, details: bool) -> ConstraintResult {
        self.0.check(puzzle, details)
    }
}
pub type NineStandardChecker = ConstraintConjunction<u8, SState<9, 9, 1, 9>, RowColChecker, NineBoxChecker>;
pub fn nine_standard_checker() -> NineStandardChecker {
    NineStandardChecker::new(RowColChecker {}, NineBoxChecker::new())
}


pub struct EightBoxChecker(BoxChecker);
impl EightBoxChecker {
    pub fn new() -> Self {
        EightBoxChecker(BoxChecker::new(4, 2, 2, 4))
    }
}
impl Constraint<u8, SState<8, 8, 1, 8>> for EightBoxChecker {
    fn check(&self, puzzle: &SState<8, 8, 1, 8>, details: bool) -> ConstraintResult {
        self.0.check(puzzle, details)
    }
}
pub type EightStandardChecker = ConstraintConjunction<u8, SState<8, 8, 1, 8>, RowColChecker, EightBoxChecker>;
pub fn eight_standard_checker() -> EightStandardChecker {
    EightStandardChecker::new(RowColChecker {}, EightBoxChecker::new())
}


pub struct SixBoxChecker(BoxChecker);
impl SixBoxChecker {
    pub fn new() -> Self {
        SixBoxChecker(BoxChecker::new(3, 2, 2, 3))
    }
}
impl Constraint<u8, SState<6, 6, 1, 6>> for SixBoxChecker {
    fn check(&self, puzzle: &SState<6, 6, 1, 6>, details: bool) -> ConstraintResult {
        self.0.check(puzzle, details)
    }
}
pub type SixStandardChecker = ConstraintConjunction<u8, SState<6, 6, 1, 6>, RowColChecker, SixBoxChecker>;
pub fn six_standard_checker() -> SixStandardChecker {
    SixStandardChecker::new(RowColChecker {}, SixBoxChecker::new())
}

pub struct FourBoxChecker(BoxChecker);
impl FourBoxChecker {
    pub fn new() -> Self {
        FourBoxChecker(BoxChecker::new(2, 2, 2, 2))
    }
}
impl Constraint<u8, SState<4, 4, 1, 4>> for FourBoxChecker {
    fn check(&self, puzzle: &SState<4, 4, 1, 4>, details: bool) -> ConstraintResult {
        self.0.check(puzzle, details)
    }
}
pub type FourStandardChecker = ConstraintConjunction<u8, SState<4, 4, 1, 4>, RowColChecker, FourBoxChecker>;
pub fn four_standard_checker() -> FourStandardChecker {
    FourStandardChecker::new(RowColChecker {}, FourBoxChecker::new())
}

pub struct FirstEmptyStrategy {}
impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Strategy<u8, SState<N, M, MIN, MAX>> for FirstEmptyStrategy {
    type ActionSet = std::vec::IntoIter<SVal<MIN, MAX>>;

    fn suggest(&self, puzzle: &SState<N, M, MIN, MAX>) -> Result<DecisionPoint<u8, SState<N, M, MIN, MAX>, Self::ActionSet>, Error> {
        for i in 0..N {
            for j in 0..M {
                if puzzle.get([i, j]).is_none() {
                    return Ok(DecisionPoint::new(
                        [i, j],
                        (MIN..=MAX).map(|value| {
                            SVal::new(value)
                        }).collect::<Vec<_>>().into_iter()
                    ));
                }
            }
        }
        Ok(DecisionPoint::empty())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::solver::FindFirstSolution;

    #[test]
    fn test_sudoku_grid() {
        let mut sudoku: SState<9, 9, 1, 9> = SState::new();
        assert_eq!(sudoku.apply([0, 0], SVal(5)), Ok(()));
        assert_eq!(sudoku.apply([8, 8], SVal(1)), Ok(()));
        assert_eq!(sudoku.get([0, 0]), Some(SVal(5)));
        assert_eq!(sudoku.undo([0, 0], SVal(5)), Ok(()));
        assert_eq!(sudoku.get([0, 0]), None);
        assert_eq!(sudoku.get([8, 8]), Some(SVal(1)));
        sudoku.reset();
        assert_eq!(sudoku.get([8, 8]), None);
    }

    #[test]
    fn test_sudoku_row_violation() {
        let mut sudoku: SState<9, 9, 1, 9> = SState::new();
        let checker = RowColChecker {};
        assert_eq!(sudoku.apply([5, 3], SVal(1)) , Ok(()));
        assert_eq!(sudoku.apply([5, 4], SVal(3)) , Ok(()));
        assert!(checker.check(&sudoku, true).is_none());
        assert_eq!(sudoku.apply([5, 8], SVal(1)) , Ok(()));
        match checker.check(&sudoku, true) {
            ConstraintResult::Details(violations) => {
                assert_eq!(violations.len(), 1);
                assert_eq!(violations[0].message, "Duplicate value 1 in row 5");
                assert_eq!(violations[0].highlight, Some(vec![[5, 8]]));
            }
            _ => panic!("Expected a detailed violation"),
        }
    }

    #[test]
    fn test_sudoku_col_violation() {
        let mut sudoku: SState<9, 9, 1, 9> = SState::new();
        let checker = RowColChecker {};
        assert_eq!(sudoku.apply([1, 3], SVal(2)) , Ok(()));
        assert_eq!(sudoku.apply([3, 3], SVal(7)) , Ok(()));
        assert!(checker.check(&sudoku, true).is_none());
        assert_eq!(sudoku.apply([6, 3], SVal(2)) , Ok(()));
        match checker.check(&sudoku, true) {
            ConstraintResult::Details(violations) => {
                assert_eq!(violations.len(), 1);
                assert_eq!(violations[0].message, "Duplicate value 2 in col 3");
                assert_eq!(violations[0].highlight, Some(vec![[6, 3]]));
            }
            _ => panic!("Expected a detailed violation"),
        }
    }

    #[test]
    fn test_sudoku_box_violation() {
        let mut sudoku: SState<9, 9, 1, 9> = SState::new();
        let checker = NineBoxChecker::new();
        assert_eq!(sudoku.apply([3, 0], SVal(8)) , Ok(()));
        assert_eq!(sudoku.apply([4, 1], SVal(2)) , Ok(()));
        assert!(checker.check(&sudoku, true).is_none());
        assert_eq!(sudoku.apply([5, 2], SVal(8)) , Ok(()));
        match checker.check(&sudoku, true) {
            ConstraintResult::Details(violations) => {
                assert_eq!(violations.len(), 1);
                assert_eq!(violations[0].message, "Duplicate value 8 in box (1, 0)");
                assert_eq!(violations[0].highlight, Some(vec![[5, 2]]));
            }
            _ => panic!("Expected a detailed violation"),
        }
    }

    #[test]
    fn test_first_empty_strategy() {
        let mut sudoku: SState<9, 9, 1, 9> = SState::new();
        let strategy = FirstEmptyStrategy {};
        assert_eq!(sudoku.apply([0, 0], SVal(3)) , Ok(()));
        assert_eq!(sudoku.apply([1, 1], SVal(3)) , Ok(()));
        let d = strategy.suggest(&sudoku).unwrap();
        assert_eq!(d.index, [0, 1]);
        assert_eq!(d.into_vec(), (1..=9).map(|v| SVal(v)).collect::<Vec<_>>());
    }

    #[test]
    fn test_sudoku_parse() {
        let input: &str = "5.3......\n\
                           6..195...\n\
                           .98....6.\n\
                           8...6...3\n\
                           4..8.3..1\n\
                           7...2...6\n\
                           .6....28.\n\
                           ...419..5\n\
                           ......8.9\n";
        let sudoku: SState<9,9, 1,9> = SState::parse(input).unwrap();
        assert_eq!(sudoku.get([0, 0]), Some(SVal::new(5)));
        assert_eq!(sudoku.get([8, 8]), Some(SVal::new(9)));
        assert_eq!(sudoku.get([2, 7]), Some(SVal::new(6)));
        assert_eq!(sudoku.to_string(), input);
    }

    #[test]
    fn test_nine_solve() {
        // #t1d1p1 from sudoku-puzzles.net
        let input: &str = ".7.583.2.\n\
                           .592..3..\n\
                           34...65.7\n\
                           795...632\n\
                           ..36971..\n\
                           68...27..\n\
                           914835.76\n\
                           .3.7.1495\n\
                           567429.13\n";
        let mut sudoku = nine_standard_parse(input).unwrap();
        let strategy = FirstEmptyStrategy {};
        let checker = nine_standard_checker();
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &strategy, &checker, false);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.get_puzzle().get([2, 2]), Some(SVal::new(2)));
                assert_eq!(solved.get_puzzle().get([2, 3]), Some(SVal::new(9)));
                assert_eq!(solved.get_puzzle().get([2, 4]), Some(SVal::new(1)));
            }
            Err(e) => panic!("Failed to solve sudoku: {:?}", e),
        }
    }

    #[test]
    fn test_eight_solve() {
        // #t34d1p1 from sudoku-puzzles.net
        let input: &str = "2...1.38\n\
                           316..7.2\n\
                           .45...8.\n\
                           1..26475\n\
                           ..475...\n\
                           52..7.6.\n\
                           .713...6\n\
                           46..8...\n";
        let mut sudoku = eight_standard_parse(input).unwrap();
        let strategy = FirstEmptyStrategy {};
        let checker = eight_standard_checker();
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &strategy, &checker, false);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.get_puzzle().get([6, 4]), Some(SVal::new(2)));
                assert_eq!(solved.get_puzzle().get([6, 5]), Some(SVal::new(5)));
                assert_eq!(solved.get_puzzle().get([6, 6]), Some(SVal::new(4)));
            }
            Err(e) => panic!("Failed to solve sudoku: {:?}", e),
        }
    }

    #[test]
    fn test_six_solve() {
        // #t2d1p1 from sudoku-puzzles.net
        let input: &str = ".3.4..\n\
                           ..56.3\n\
                           ...1..\n\
                           .1.3.5\n\
                           .64.31\n\
                           ..1.46\n";
        let mut sudoku = six_standard_parse(input).unwrap();
        let strategy = FirstEmptyStrategy {};
        let checker = six_standard_checker();
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &strategy, &checker, false);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.get_puzzle().get([2, 0]), Some(SVal::new(6)));
                assert_eq!(solved.get_puzzle().get([2, 1]), Some(SVal::new(5)));
                assert_eq!(solved.get_puzzle().get([2, 2]), Some(SVal::new(3)));
            }
            Err(e) => panic!("Failed to solve sudoku: {:?}", e),
        }
    }

    #[test]
    fn test_four_solve() {
        // #t14d1p1 from sudoku-puzzles.net
        let input: &str = "...4\n\
                           ....\n\
                           2..3\n\
                           4.12\n";
        let mut sudoku = four_standard_parse(input).unwrap();
        let strategy = FirstEmptyStrategy {};
        let checker = four_standard_checker();
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &strategy, &checker, false);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.get_puzzle().get([0, 0]), Some(SVal::new(1)));
                assert_eq!(solved.get_puzzle().get([0, 1]), Some(SVal::new(2)));
                assert_eq!(solved.get_puzzle().get([0, 2]), Some(SVal::new(3)));
            }
            Err(e) => panic!("Failed to solve sudoku: {:?}", e),
        }
    }
}