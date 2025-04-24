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

// TODO Rectangle checker

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

    // TODO: Add more tests for the constraints and strategies
}