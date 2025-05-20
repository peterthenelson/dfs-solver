use std::fmt::Debug;
use crate::core::{State, Index, UInt};

/// Potential violation of a constraint. The optimized approach to use in the
/// solver is to return the first violation found without any detailed
/// information. However, for debugging or UI purposes, it may be useful to
/// return a list of all the violations, along with the specific actions that
/// caused them.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintResult {
    Simple(&'static str),
    Details(Vec<ConstraintViolationDetail>),
    NoViolation,
}

impl ConstraintResult {
    pub fn is_none(&self) -> bool {
        match self {
            ConstraintResult::NoViolation => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintViolationDetail {
    pub message: String,
    pub highlight: Option<Vec<Index>>,
}

/// Trait for checking that the current solve-state is valid.
pub trait Constraint<U: UInt, P: State<U>> {
    fn check(&self, puzzle: &P, details: bool) -> ConstraintResult;
}

pub struct ConstraintConjunction<U, P, X, Y>
where
    U: UInt, P: State<U>, X: Constraint<U, P>, Y: Constraint<U, P>
{
    pub x: X,
    pub y: Y,
    pub p_u: std::marker::PhantomData<U>,
    pub p_p: std::marker::PhantomData<P>,
}

impl <U, P, X, Y> ConstraintConjunction<U, P, X, Y>
where
    U: UInt, P: State<U>, X: Constraint<U, P>, Y: Constraint<U, P>
{
    pub fn new(x: X, y: Y) -> Self {
        ConstraintConjunction { x, y, p_u: std::marker::PhantomData, p_p: std::marker::PhantomData }
    }
}

impl <U, P, X, Y> Constraint<U, P> for ConstraintConjunction<U, P, X, Y>
where
    U: UInt, P: State<U>, X: Constraint<U, P>, Y: Constraint<U, P>
{
    fn check(&self, puzzle: &P, details: bool) -> ConstraintResult {
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
    use crate::core::{to_value, Error, State, UVGrid, UVal, UVUnwrapped, UVWrapped, Value};

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct Val(pub u8);
    impl Value<u8> for Val {
        fn parse(_: &str) -> Result<Self, Error> { todo!() }
        fn cardinality() -> usize { 9 }
        fn from_uval(u: UVal<u8, UVUnwrapped>) -> Self { Val(u.value()) }
        fn to_uval(self) -> UVal<u8, UVWrapped> { UVal::new(self.0) }
    }

    #[derive(Debug, Clone)]
    pub struct ThreeVals {
        pub grid: UVGrid<u8>,
    }
    impl State<u8> for ThreeVals {
        type Value = Val;
        const ROWS: usize = 1;
        const COLS: usize = 3;

        fn reset(&mut self) { self.grid = UVGrid::new(Self::ROWS, Self::COLS); }
        fn get(&self, index: Index) -> Option<Self::Value> { self.grid.get(index).map(to_value) }
        fn apply(&mut self, index: Index, value: Self::Value) -> Result<(), Error> {
            self.grid.set(index, Some(value.to_uval()));
            Ok(())
        }
        fn undo(&mut self, index: Index, _: Self::Value) -> Result<(), Error> {
            self.grid.set(index, None);
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    pub struct BlacklistedVal(pub u8);
    impl Constraint<u8, ThreeVals> for BlacklistedVal {
        fn check(&self, puzzle: &ThreeVals, _details: bool) -> ConstraintResult {
            for j in 0..3 {
                if puzzle.grid.get([0, j]).map(to_value) == Some(Val(self.0)) {
                    return ConstraintResult::Simple("Blacklisted value found");
                }
            }
            ConstraintResult::NoViolation
        }
    }

    #[test]
    fn test_constraint_conjunction() {
        let mut puzzle = ThreeVals { grid: UVGrid::new(ThreeVals::ROWS, ThreeVals::COLS) };
        let constraint1 = BlacklistedVal(1);
        let constraint2 = BlacklistedVal(2);
        let conjunction = ConstraintConjunction::new(constraint1, constraint2);
        assert_eq!(conjunction.check(&puzzle, false), ConstraintResult::NoViolation);
        puzzle.apply([0, 0], Val(1)).unwrap();
        assert_eq!(conjunction.check(&puzzle, false), ConstraintResult::Simple("Blacklisted value found"));
        puzzle.apply([0, 0], Val(3)).unwrap();
        assert_eq!(conjunction.check(&puzzle, false), ConstraintResult::NoViolation);
        puzzle.apply([0, 1], Val(2)).unwrap();
        assert_eq!(conjunction.check(&puzzle, false), ConstraintResult::Simple("Blacklisted value found"));
    }
}