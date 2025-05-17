use std::fmt::Debug;
use crate::puzzle::{PuzzleState, PuzzleIndex, UInt};

/// Potential violation of a constraint. The optimized approach to use in the
/// solver is to return the first violation found without any detailed
/// information. However, for debugging or UI purposes, it may be useful to
/// return a list of all the violations, along with the specific actions that
/// caused them.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintResult<const DIM: usize> {
    Simple(&'static str),
    Details(Vec<ConstraintViolationDetail<DIM>>),
    NoViolation,
}

impl <const DIM: usize> ConstraintResult<DIM> {
    pub fn is_none(&self) -> bool {
        match self {
            ConstraintResult::NoViolation => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintViolationDetail<const DIM: usize> {
    pub message: String,
    pub highlight: Option<Vec<PuzzleIndex<DIM>>>,
}

/// Trait for checking that the current solve-state is valid.
pub trait Constraint<const DIM: usize, U: UInt, P: PuzzleState<DIM, U>> {
    fn check(&self, puzzle: &P, details: bool) -> ConstraintResult<DIM>;
}

pub struct ConstraintConjunction<const DIM: usize, U, P, X, Y>
where
    U: UInt, P: PuzzleState<DIM, U>, X: Constraint<DIM, U, P>, Y: Constraint<DIM, U, P>
{
    pub x: X,
    pub y: Y,
    pub p_u: std::marker::PhantomData<U>,
    pub p_p: std::marker::PhantomData<P>,
}

impl <const DIM: usize, U, P, X, Y> ConstraintConjunction<DIM, U, P, X, Y>
where
    U: UInt, P: PuzzleState<DIM, U>, X: Constraint<DIM, U, P>, Y: Constraint<DIM, U, P>
{
    pub fn new(x: X, y: Y) -> Self {
        ConstraintConjunction { x, y, p_u: std::marker::PhantomData, p_p: std::marker::PhantomData }
    }
}

impl <const DIM: usize, U, P, X, Y> Constraint<DIM, U, P> for ConstraintConjunction<DIM, U, P, X, Y>
where
    U: UInt, P: PuzzleState<DIM, U>, X: Constraint<DIM, U, P>, Y: Constraint<DIM, U, P>
{
    fn check(&self, puzzle: &P, details: bool) -> ConstraintResult<DIM> {
        if details {
            let mut violations = Vec::new();
            match self.x.check(puzzle, details) {
                ConstraintResult::NoViolation => (),
                ConstraintResult::Simple(message) => {
                    violations.push(ConstraintViolationDetail {
                        message: message.to_string(),
                        highlight: None,
                    });
                },
                ConstraintResult::Details(mut violation) => {
                    violations.append(&mut violation);
                }
            };
            match self.y.check(puzzle, details) {
                ConstraintResult::NoViolation => (),
                ConstraintResult::Simple(message) => {
                    violations.push(ConstraintViolationDetail {
                        message: message.to_string(),
                        highlight: None,
                    });
                },
                ConstraintResult::Details(mut violation) => {
                    violations.append(&mut violation);
                }
            };
            if violations.is_empty() {
                return ConstraintResult::NoViolation;
            }
            ConstraintResult::Details(violations)
        } else {
            match self.x.check(puzzle, details) {
                ConstraintResult::NoViolation => self.y.check(puzzle, details),
                x_result => x_result,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle::{PuzzleError, PuzzleState, PuzzleValue};

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct Val(pub u8);
    impl PuzzleValue<u8> for Val {
        fn cardinality() -> usize { 9 }
        fn from_uint(u: u8) -> Self { Val(u) }
        fn to_uint(self) -> u8 { self.0 }
    }

    #[derive(Debug, Clone)]
    pub struct ThreeVals {
        pub grid: [Option<Val>; 3]
    }
    impl PuzzleState<1, u8> for ThreeVals {
        type Value = Val;
        fn reset(&mut self) { self.grid = [None; 3]; }
        fn get(&self, index: PuzzleIndex<1>) -> Option<Self::Value> { self.grid[index[0]] }
        fn apply(&mut self, index: PuzzleIndex<1>, value: Self::Value) -> Result<(), PuzzleError> {
            self.grid[index[0]] = Some(value);
            Ok(())
        }
        fn undo(&mut self, index: PuzzleIndex<1>) -> Result<(), PuzzleError> {
            self.grid[index[0]] = None;
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    pub struct BlacklistedVal(pub u8);
    impl Constraint<1, u8, ThreeVals> for BlacklistedVal {
        fn check(&self, puzzle: &ThreeVals, _details: bool) -> ConstraintResult<1> {
            if puzzle.grid.iter().any(|&v| v == Some(Val(self.0))) {
                ConstraintResult::Simple("Blacklisted value found")
            } else {
                ConstraintResult::NoViolation
            }
        }
    }

    #[test]
    fn test_constraint_conjunction() {
        let mut puzzle = ThreeVals { grid: [None, None, None] };
        let constraint1 = BlacklistedVal(1);
        let constraint2 = BlacklistedVal(2);
        let conjunction = ConstraintConjunction::new(constraint1, constraint2);
        assert_eq!(conjunction.check(&puzzle, false), ConstraintResult::NoViolation);
        puzzle.apply([0], Val(1)).unwrap();
        assert_eq!(conjunction.check(&puzzle, false), ConstraintResult::Simple("Blacklisted value found"));
        puzzle.apply([0], Val(3)).unwrap();
        assert_eq!(conjunction.check(&puzzle, false), ConstraintResult::NoViolation);
        puzzle.apply([1], Val(2)).unwrap();
        assert_eq!(conjunction.check(&puzzle, false), ConstraintResult::Simple("Blacklisted value found"));
    }
}