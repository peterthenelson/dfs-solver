use std::fmt::Display;
use crate::dfs::{Constraint, ConstraintViolation, PuzzleError, PuzzleState, Strategy};

#[derive(Debug, Clone, PartialEq)]
pub struct SudokuAction<const MIN: u8 = 1, const MAX: u8 = 9> {
    pub row: usize,
    pub col: usize,
    pub value: u8,
}

#[derive(Debug, Clone)]
pub struct Sudoku<const N: usize = 9, const M: usize = 9, const MIN: u8 = 1, const MAX: u8 = 9> {
    pub grid: [[Option<u8>; M]; N],
} 

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Sudoku<N, M, MIN, MAX> {
    pub fn parse(s: &str) -> Result<Self, PuzzleError> {
        let mut grid = [[None; M]; N];
        let lines: Vec<&str> = s.lines().collect();
        if lines.len() != N {
            return Err(PuzzleError::new("Invalid number of rows".to_string()));
        }
        for i in 0..N {
            let line = lines[i].trim();
            if line.len() != M {
                return Err(PuzzleError::new("Invalid number of columns".to_string()));
            }
            for j in 0..M {
                let c = line.chars().nth(j).unwrap();
                if c == '.' {
                    grid[i][j] = None;
                } else if c.is_digit(10) {
                    grid[i][j] = Some(c.to_digit(10).unwrap() as u8);
                } else {
                    return Err(PuzzleError::new("Invalid character in input".to_string()));
                }
            }
        }
        Ok(Self { grid })
    }

    pub fn serialize(&self) -> String {
        let mut result = String::new();
        for row in &self.grid {
            for &cell in row {
                if let Some(value) = cell {
                    result.push_str(&value.to_string());
                } else {
                    result.push('.');
                }
            }
            result.push('\n');
        }
        result
    }
}

pub fn nine_standard_parse(s: &str) -> Result<Sudoku<9, 9, 1, 9>, PuzzleError> {
    Sudoku::<9, 9, 1, 9>::parse(s)
}

pub fn eight_standard_parse(s: &str) -> Result<Sudoku<8, 8, 1, 8>, PuzzleError> {
    Sudoku::<8, 8, 1, 8>::parse(s)
}

pub fn six_standard_parse(s: &str) -> Result<Sudoku<6, 6, 1, 6>, PuzzleError> {
    Sudoku::<6, 6, 1, 6>::parse(s)
}

pub fn four_standard_parse(s: &str) -> Result<Sudoku<4, 4, 1, 4>, PuzzleError> {
    Sudoku::<4, 4, 1, 4>::parse(s)
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Display for Sudoku<N, M, MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        for row in &self.grid {
            for &cell in row {
                if let Some(value) = cell {
                    write!(f, "{}", value)?;
                } else {
                    write!(f, ".")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub const OUT_OF_BOUNDS_ERROR: PuzzleError = PuzzleError::new_const("Out of bounds");
pub const ALREADY_FILLED_ERROR: PuzzleError = PuzzleError::new_const("Cell already filled");
pub const NO_SUCH_ACTION_ERROR: PuzzleError = PuzzleError::new_const("No such action to undo");

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> PuzzleState<SudokuAction<MIN, MAX>> for Sudoku<N, M, MIN, MAX> {
    fn apply(&mut self, action: &SudokuAction<MIN, MAX>) -> Result<(), PuzzleError> {
        if action.row >= N || action.col >= M {
            return Err(OUT_OF_BOUNDS_ERROR);
        }
        if self.grid[action.row][action.col].is_some() {
            return Err(ALREADY_FILLED_ERROR);
        }
        self.grid[action.row][action.col] = Some(action.value);
        Ok(())
    }

    fn undo(&mut self, action: &SudokuAction<MIN, MAX>) -> Result<(), PuzzleError> {
        if action.row >= N || action.col >= M {
            return Err(OUT_OF_BOUNDS_ERROR);
        }
        if self.grid[action.row][action.col].is_none() {
            return Err(NO_SUCH_ACTION_ERROR);
        }
        self.grid[action.row][action.col] = None;
        Ok(())
    }

    fn reset(&mut self) {
        self.grid = [[None; M]; N];
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Sudoku<N, M, MIN, MAX> {
    pub fn new() -> Self {
        Self { grid: [[None; M]; N] }
    }

    // TODO: Loading and saving to a string
}

pub struct RowColChecker {}
impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Constraint<SudokuAction<MIN, MAX>, Sudoku<N, M, MIN, MAX>> for RowColChecker {
    fn check(&self, puzzle: &Sudoku<N, M, MIN, MAX>, details: bool) -> Option<ConstraintViolation<SudokuAction<MIN, MAX>>> {
        for r in 0..N {
            let mut seen = vec![false; (MAX - MIN + 1) as usize];
            for c in 0..M {
                if let Some(value) = puzzle.grid[r][c] {
                    if seen[(value - MIN) as usize] {
                        let highlight: Option<Vec<SudokuAction<MIN, MAX>>> = if details {
                            Some(vec![SudokuAction { row: r, col: c, value }])
                        } else {
                            None
                        };
                        return Some(ConstraintViolation {
                            message: format!("Duplicate value {} in row {}", value, r),
                            highlight,
                        });
                    }
                    seen[(value - MIN) as usize] = true;
                }
            }
        }
        for c in 0..M {
            let mut seen = vec![false; (MAX - MIN + 1) as usize];
            for r in 0..N {
                if let Some(value) = puzzle.grid[r][c] {
                    if seen[(value - MIN) as usize] {
                        let highlight: Option<Vec<SudokuAction<MIN, MAX>>> = if details {
                            Some(vec![SudokuAction { row: r, col: c, value }])
                        } else {
                            None
                        };
                        return Some(ConstraintViolation {
                            message: format!("Duplicate value {} in col {}", value, c),
                            highlight,
                        });
                    }
                    seen[(value - MIN) as usize] = true;
                }
            }
        }
        None
    }
}

pub struct NineBoxChecker {}
impl <const MIN: u8, const MAX: u8> Constraint<SudokuAction<MIN, MAX>, Sudoku<9, 9, MIN, MAX>> for NineBoxChecker {
    fn check(&self, puzzle: &Sudoku<9, 9, MIN, MAX>, details: bool) -> Option<ConstraintViolation<SudokuAction<MIN, MAX>>> {
        for box_row in 0..3 {
            for box_col in 0..3 {
                let mut seen = vec![false; (MAX - MIN + 1) as usize];
                for r in 0..3 {
                    for c in 0..3 {
                        let row = box_row * 3 + r;
                        let col = box_col * 3 + c;
                        if let Some(value) = puzzle.grid[row][col] {
                            if seen[(value - MIN) as usize] {
                                let highlight: Option<Vec<SudokuAction<MIN, MAX>>> = if details {
                                    Some(vec![SudokuAction { row, col, value }])
                                } else {
                                    None
                                };
                                return Some(ConstraintViolation {
                                    message: format!("Duplicate value {} in box ({}, {})", value, box_row, box_col),
                                    highlight,
                                });
                            }
                            seen[(value - MIN) as usize] = true;
                        }
                    }
                }
            }
        }
        None
    }
}

pub struct NineStandardChecker { pub row_col_checker: RowColChecker, pub box_checker: NineBoxChecker }

impl NineStandardChecker {
    pub fn new() -> Self {
        Self { row_col_checker: RowColChecker {}, box_checker: NineBoxChecker {} }
    }
}

impl <const MIN: u8, const MAX: u8> Constraint<SudokuAction<MIN, MAX>, Sudoku<9, 9, MIN, MAX>> for NineStandardChecker {
    fn check(&self, puzzle: &Sudoku<9, 9, MIN, MAX>, details: bool) -> Option<ConstraintViolation<SudokuAction<MIN, MAX>>> {
        let row_col_violation = self.row_col_checker.check(puzzle, details);
        if row_col_violation.is_some() {
            return row_col_violation;
        }
        self.box_checker.check(puzzle, details)
    }
}

pub struct EightBoxChecker {}
impl <const MIN: u8, const MAX: u8> Constraint<SudokuAction<MIN, MAX>, Sudoku<8, 8, MIN, MAX>> for EightBoxChecker {
    fn check(&self, puzzle: &Sudoku<8, 8, MIN, MAX>, details: bool) -> Option<ConstraintViolation<SudokuAction<MIN, MAX>>> {
        for box_row in 0..4 {
            for box_col in 0..2 {
                let mut seen = vec![false; (MAX - MIN + 1) as usize];
                for r in 0..2 {
                    for c in 0..4 {
                        let row = box_row * 2 + r;
                        let col = box_col * 4 + c;
                        if let Some(value) = puzzle.grid[row][col] {
                            if seen[(value - MIN) as usize] {
                                let highlight: Option<Vec<SudokuAction<MIN, MAX>>> = if details {
                                    Some(vec![SudokuAction { row, col, value }])
                                } else {
                                    None
                                };
                                return Some(ConstraintViolation {
                                    message: format!("Duplicate value {} in box ({}, {})", value, box_row, box_col),
                                    highlight,
                                });
                            }
                            seen[(value - MIN) as usize] = true;
                        }
                    }
                }
            }
        }
        None
    }
}
pub struct EightStandardChecker { pub row_col_checker: RowColChecker, pub box_checker: EightBoxChecker }

impl EightStandardChecker {
    pub fn new() -> Self {
        Self { row_col_checker: RowColChecker {}, box_checker: EightBoxChecker {} }
    }
}

impl <const MIN: u8, const MAX: u8> Constraint<SudokuAction<MIN, MAX>, Sudoku<8, 8, MIN, MAX>> for EightStandardChecker {
    fn check(&self, puzzle: &Sudoku<8, 8, MIN, MAX>, details: bool) -> Option<ConstraintViolation<SudokuAction<MIN, MAX>>> {
        let row_col_violation = self.row_col_checker.check(puzzle, details);
        if row_col_violation.is_some() {
            return row_col_violation;
        }
        self.box_checker.check(puzzle, details)
    }
}

pub struct SixBoxChecker {}
impl <const MIN: u8, const MAX: u8> Constraint<SudokuAction<MIN, MAX>, Sudoku<6, 6, MIN, MAX>> for SixBoxChecker {
    fn check(&self, puzzle: &Sudoku<6, 6, MIN, MAX>, details: bool) -> Option<ConstraintViolation<SudokuAction<MIN, MAX>>> {
        for box_row in 0..3 {
            for box_col in 0..2 {
                let mut seen = vec![false; (MAX - MIN + 1) as usize];
                for r in 0..2 {
                    for c in 0..3 {
                        let row = box_row * 2 + r;
                        let col = box_col * 3 + c;
                        if let Some(value) = puzzle.grid[row][col] {
                            if seen[(value - MIN) as usize] {
                                let highlight: Option<Vec<SudokuAction<MIN, MAX>>> = if details {
                                    Some(vec![SudokuAction { row, col, value }])
                                } else {
                                    None
                                };
                                return Some(ConstraintViolation {
                                    message: format!("Duplicate value {} in box ({}, {})", value, box_row, box_col),
                                    highlight,
                                });
                            }
                            seen[(value - MIN) as usize] = true;
                        }
                    }
                }
            }
        }
        None
    }
}
pub struct SixStandardChecker { pub row_col_checker: RowColChecker, pub box_checker: SixBoxChecker }

impl SixStandardChecker {
    pub fn new() -> Self {
        Self { row_col_checker: RowColChecker {}, box_checker: SixBoxChecker {} }
    }
}

impl <const MIN: u8, const MAX: u8> Constraint<SudokuAction<MIN, MAX>, Sudoku<6, 6, MIN, MAX>> for SixStandardChecker {
    fn check(&self, puzzle: &Sudoku<6, 6, MIN, MAX>, details: bool) -> Option<ConstraintViolation<SudokuAction<MIN, MAX>>> {
        let row_col_violation = self.row_col_checker.check(puzzle, details);
        if row_col_violation.is_some() {
            return row_col_violation;
        }
        self.box_checker.check(puzzle, details)
    }
}

pub struct FourBoxChecker {}
impl <const MIN: u8, const MAX: u8> Constraint<SudokuAction<MIN, MAX>, Sudoku<4, 4, MIN, MAX>> for FourBoxChecker {
    fn check(&self, puzzle: &Sudoku<4, 4, MIN, MAX>, details: bool) -> Option<ConstraintViolation<SudokuAction<MIN, MAX>>> {
        for box_row in 0..2 {
            for box_col in 0..2 {
                let mut seen = vec![false; (MAX - MIN + 1) as usize];
                for r in 0..2 {
                    for c in 0..2 {
                        let row = box_row * 2 + r;
                        let col = box_col * 2 + c;
                        if let Some(value) = puzzle.grid[row][col] {
                            if seen[(value - MIN) as usize] {
                                let highlight: Option<Vec<SudokuAction<MIN, MAX>>> = if details {
                                    Some(vec![SudokuAction { row, col, value }])
                                } else {
                                    None
                                };
                                return Some(ConstraintViolation {
                                    message: format!("Duplicate value {} in box ({}, {})", value, box_row, box_col),
                                    highlight,
                                });
                            }
                            seen[(value - MIN) as usize] = true;
                        }
                    }
                }
            }
        }
        None
    }
}
pub struct FourStandardChecker { pub row_col_checker: RowColChecker, pub box_checker: FourBoxChecker }

impl FourStandardChecker {
    pub fn new() -> Self {
        Self { row_col_checker: RowColChecker {}, box_checker: FourBoxChecker {} }
    }
}

impl <const MIN: u8, const MAX: u8> Constraint<SudokuAction<MIN, MAX>, Sudoku<4, 4, MIN, MAX>> for FourStandardChecker {
    fn check(&self, puzzle: &Sudoku<4, 4, MIN, MAX>, details: bool) -> Option<ConstraintViolation<SudokuAction<MIN, MAX>>> {
        let row_col_violation = self.row_col_checker.check(puzzle, details);
        if row_col_violation.is_some() {
            return row_col_violation;
        }
        self.box_checker.check(puzzle, details)
    }
}

pub struct FirstEmptyStrategy {}
impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Strategy<SudokuAction<MIN, MAX>, Sudoku<N, M, MIN, MAX>> for FirstEmptyStrategy {
    type ActionSet = std::vec::IntoIter<SudokuAction<MIN, MAX>>;

    fn suggest(&self, puzzle: &Sudoku<N, M, MIN, MAX>) -> Result<Self::ActionSet, PuzzleError> {
        for i in 0..N {
            for j in 0..M {
                if puzzle.grid[i][j].is_none() {
                    return Ok((MIN..=MAX).map(|value| {
                        SudokuAction { row: i, col: j, value }
                    }).collect::<Vec<_>>().into_iter());
                }
            }
        }
        let empty: Self::ActionSet = vec![].into_iter();
        Ok(empty)
    }

    fn empty_action_set() -> Self::ActionSet {
        vec![].into_iter()
    }
}

#[cfg(test)]
mod test {
    use crate::dfs::FindFirstSolution;

    use super::*;

    #[test]
    fn test_sudoku_grid() {
        let mut sudoku: Sudoku<9, 9, 1, 9> = Sudoku::new();
        let action: SudokuAction<1, 9> = SudokuAction { row: 0, col: 0, value: 5 };
        let action2: SudokuAction<1, 9> = SudokuAction { row: 8, col: 8, value: 1 };
        assert_eq!(sudoku.apply(&action), Ok(()));
        assert_eq!(sudoku.apply(&action2), Ok(()));
        assert_eq!(sudoku.grid[0][0], Some(5));
        assert_eq!(sudoku.undo(&action), Ok(()));
        assert_eq!(sudoku.grid[0][0], None);
        assert_eq!(sudoku.grid[8][8], Some(1));
        sudoku.reset();
        assert_eq!(sudoku.grid[8][8], None);
    }

    #[test]
    fn test_sudoku_row_violation() {
        let mut sudoku: Sudoku<9, 9, 1, 9> = Sudoku::new();
        let checker = RowColChecker {};
        assert_eq!(sudoku.apply(&SudokuAction { row: 5, col: 3, value: 1 }), Ok(()));
        assert_eq!(sudoku.apply(&SudokuAction { row: 5, col: 4, value: 3 }), Ok(()));
        assert!(checker.check(&sudoku, true).is_none());
        assert_eq!(sudoku.apply(&SudokuAction { row: 5, col: 8, value: 1 }), Ok(()));
        match checker.check(&sudoku, true) {
            Some(ConstraintViolation { message, highlight }) => {
                assert_eq!(message, "Duplicate value 1 in row 5");
                assert_eq!(highlight, Some(vec![SudokuAction { row: 5, col: 8, value: 1 }]));
            }
            None => panic!("Expected a violation"),
        }
    }

    #[test]
    fn test_sudoku_col_violation() {
        let mut sudoku: Sudoku<9, 9, 1, 9> = Sudoku::new();
        let checker = RowColChecker {};
        assert_eq!(sudoku.apply(&SudokuAction { row: 1, col: 3, value: 2 }), Ok(()));
        assert_eq!(sudoku.apply(&SudokuAction { row: 3, col: 3, value: 7 }), Ok(()));
        assert!(checker.check(&sudoku, true).is_none());
        assert_eq!(sudoku.apply(&SudokuAction { row: 6, col: 3, value: 2 }), Ok(()));
        match checker.check(&sudoku, true) {
            Some(ConstraintViolation { message, highlight }) => {
                assert_eq!(message, "Duplicate value 2 in col 3");
                assert_eq!(highlight, Some(vec![SudokuAction { row: 6, col: 3, value: 2 }]));
            }
            None => panic!("Expected a violation"),
        }
    }

    #[test]
    fn test_sudoku_box_violation() {
        let mut sudoku: Sudoku<9, 9, 1, 9> = Sudoku::new();
        let checker = NineBoxChecker {};
        assert_eq!(sudoku.apply(&SudokuAction { row: 3, col: 0, value: 8 }), Ok(()));
        assert_eq!(sudoku.apply(&SudokuAction { row: 4, col: 1, value: 2 }), Ok(()));
        assert!(checker.check(&sudoku, true).is_none());
        assert_eq!(sudoku.apply(&SudokuAction { row: 5, col: 2, value: 8 }), Ok(()));
        match checker.check(&sudoku, true) {
            Some(ConstraintViolation { message, highlight }) => {
                assert_eq!(message, "Duplicate value 8 in box (1, 0)");
                assert_eq!(highlight, Some(vec![SudokuAction { row: 5, col: 2, value: 8 }]));
            }
            None => panic!("Expected a violation"),
        }
    }

    #[test]
    fn test_first_empty_strategy() {
        let mut sudoku: Sudoku<9, 9, 1, 9> = Sudoku::new();
        let strategy = FirstEmptyStrategy {};
        assert_eq!(sudoku.apply(&SudokuAction { row: 0, col: 0, value: 5 }), Ok(()));
        assert_eq!(sudoku.apply(&SudokuAction { row: 1, col: 1, value: 3 }), Ok(()));
        let mut action_set = strategy.suggest(&sudoku).unwrap();
        match action_set.next() {
            Some(SudokuAction { row, col, value }) => {
                assert_eq!(row, 0);
                assert_eq!(col, 1);
                assert_eq!(value, 1);
            }
            None => panic!("Expected an action"),
        }
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
        let sudoku: Sudoku<9,9, 1,9> = Sudoku::parse(input).unwrap();
        assert_eq!(sudoku.grid[0][0], Some(5));
        assert_eq!(sudoku.grid[8][8], Some(9));
        assert_eq!(sudoku.grid[2][7], Some(6));
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
        let checker = NineStandardChecker::new();
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &strategy, vec![&checker], false);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.get_puzzle().grid[2][2], Some(2));
                assert_eq!(solved.get_puzzle().grid[2][3], Some(9));
                assert_eq!(solved.get_puzzle().grid[2][4], Some(1));
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
        let checker = EightStandardChecker::new();
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &strategy, vec![&checker], false);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.get_puzzle().grid[6][4], Some(2));
                assert_eq!(solved.get_puzzle().grid[6][5], Some(5));
                assert_eq!(solved.get_puzzle().grid[6][6], Some(4));
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
        let checker = SixStandardChecker::new();
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &strategy, vec![&checker], false);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.get_puzzle().grid[2][0], Some(6));
                assert_eq!(solved.get_puzzle().grid[2][1], Some(5));
                assert_eq!(solved.get_puzzle().grid[2][2], Some(3));
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
        let checker = FourStandardChecker::new();
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &strategy, vec![&checker], false);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.get_puzzle().grid[0][0], Some(1));
                assert_eq!(solved.get_puzzle().grid[0][1], Some(2));
                assert_eq!(solved.get_puzzle().grid[0][2], Some(3));
            }
            Err(e) => panic!("Failed to solve sudoku: {:?}", e),
        }
    }
}