use std::fmt::Debug;
use crate::core::{ConstraintResult, DecisionGrid, Error, Index, State, Stateful, UInt};

/// A full explanation of a violated constraint (for UI and debugging purposes).
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintViolationDetail {
    pub message: String,
    pub highlight: Option<Vec<Index>>,
}

/// Constraints check that the puzzle state is valid. The ideal Constraint 
/// will:
/// - Return early when it hits a Contradiction or Certainty.
/// - Be able to provide a useful explanation (for UI or debugging purposes) for
///   any contradictions.
/// - Be able to usefully update a full DecisionGrid (possibly with ranking
///   features) when neither a Contradiction nor a Certainty has been found.
/// - Keep its internal state updated by implementing Stateful so that the grid
///   computations are not costly.
/// 
/// However, it's acceptable to use this API for other use-cases as well:
/// 1. Some constraints may not be well-adapted to usefully updating the cells
///    in a DecisionGrid. E.g., maybe there's a parity constraint for the sum of
///    a box or row or w/e. In these cases, it's certainly legal to implement a
///    constraint that only ever returns a Contradiction (if found) or Ok
///    (otherwise). It just makes the solver's work more difficult.
/// 2. Sometimes you may have a deductive rule for determining the value of
///    cells that doesn't seem like a Constraint at all. It's find to also
///    implement these as Constraints that only ever return a Certainty (if it
///    can be deduced) or Ok (otherwise).
pub trait Constraint<U: UInt, S: State<U>> where Self: Stateful<U, S::Value> + Debug {
    /// Check that the Constraint is satisfied by the puzzle (and any internal
    /// state from past actions). If a Constraint is able to infer useful
    /// information about what values a cell could take on, they should update
    /// the provided grid (in a way that further constrains it).
    fn check(&self, puzzle: &S, grid: &mut DecisionGrid<U, S::Value>) -> ConstraintResult<U, S::Value>;
    fn explain_contradictions(&self, puzzle: &S) -> Vec<ConstraintViolationDetail>;
}

pub struct ConstraintConjunction<U, S, X, Y>
where
    U: UInt, S: State<U>, X: Constraint<U, S>, Y: Constraint<U, S>
{
    pub x: X,
    pub y: Y,
    pub _p_u: std::marker::PhantomData<U>,
    pub _p_s: std::marker::PhantomData<S>,
}

impl <U, S, X, Y> ConstraintConjunction<U, S, X, Y>
where
    U: UInt, S: State<U>, X: Constraint<U, S>, Y: Constraint<U, S>
{
    pub fn new(x: X, y: Y) -> Self {
        ConstraintConjunction { x, y, _p_u: std::marker::PhantomData, _p_s: std::marker::PhantomData }
    }
}

impl <U, S, X, Y> Debug for ConstraintConjunction<U, S, X, Y>
where
    U: UInt, S: State<U>, X: Constraint<U, S>, Y: Constraint<U, S>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}{:?}", self.x, self.y)
    }
}

impl <U, S, X, Y> Stateful<U, S::Value> for ConstraintConjunction<U, S, X, Y>
where
    U: UInt, S: State<U>, X: Constraint<U, S>, Y: Constraint<U, S> {
    fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
    }

    fn apply(&mut self, index: Index, value: S::Value) -> Result<(), Error> {
        let xres = self.x.apply(index, value);
        let yres = self.y.apply(index, value);
        if xres.is_err() { xres } else { yres }
    }

    fn undo(&mut self, index: Index, value: S::Value) -> Result<(), Error> {
        let xres = self.x.undo(index, value);
        let yres = self.y.undo(index, value);
        if xres.is_err() { xres } else { yres }
    }
}

impl <U, S, X, Y> Constraint<U, S> for ConstraintConjunction<U, S, X, Y>
where
    U: UInt, S: State<U>, X: Constraint<U, S>, Y: Constraint<U, S>
{
    fn check(&self, puzzle: &S, grid: &mut DecisionGrid<U, S::Value>) -> ConstraintResult<U, S::Value> {
        match self.x.check(puzzle, grid) {
            ConstraintResult::Contradiction => ConstraintResult::Contradiction,
            ConstraintResult::Certainty(d) => ConstraintResult::Certainty(d),
            ConstraintResult::Ok => self.y.check(puzzle, grid),
        }
    }

    fn explain_contradictions(&self, puzzle: &S) -> Vec<ConstraintViolationDetail> {
        let mut violations = Vec::new();
        violations.extend(self.x.explain_contradictions(puzzle));
        violations.extend(self.y.explain_contradictions(puzzle));
        violations
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::{singleton_set, to_value, unpack_values, DecisionGrid, Error, State, Stateful, UVGrid, UVUnwrapped, UVWrapped, UVal, Value};

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct Val(pub u8);
    impl Value<u8> for Val {
        fn parse(_: &str) -> Result<Self, Error> { todo!() }
        fn cardinality() -> usize { 9 }
        fn possiblities() -> Vec<Self> { (1..=9).map(Val).collect() }
        fn from_uval(u: UVal<u8, UVUnwrapped>) -> Self { Val(u.value()+1) }
        fn to_uval(self) -> UVal<u8, UVWrapped> { UVal::new(self.0-1) }
    }

    #[derive(Debug, Clone)]
    pub struct ThreeVals {
        pub grid: UVGrid<u8>,
    }
    impl State<u8> for ThreeVals {
        type Value = Val;
        const ROWS: usize = 1;
        const COLS: usize = 3;
        fn get(&self, index: Index) -> Option<Self::Value> { self.grid.get(index).map(to_value) }
    }
    impl Stateful<u8, Val> for ThreeVals {
        fn reset(&mut self) { self.grid = UVGrid::new(Self::ROWS, Self::COLS); }
        fn apply(&mut self, index: Index, value: Val) -> Result<(), Error> {
            self.grid.set(index, Some(value.to_uval()));
            Ok(())
        }
        fn undo(&mut self, index: Index, _: Val) -> Result<(), Error> {
            self.grid.set(index, None);
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    pub struct BlacklistedVal(pub u8);
    impl Stateful<u8, Val> for BlacklistedVal {}
    impl Constraint<u8, ThreeVals> for BlacklistedVal {
        fn check(&self, puzzle: &ThreeVals, grid: &mut DecisionGrid<u8, Val>) -> ConstraintResult<u8, Val> {
            for j in 0..3 {
                if puzzle.grid.get([0, j]).map(to_value) == Some(Val(self.0)) {
                    return ConstraintResult::Contradiction;
                } else {
                    grid.get_mut([0, j]).0.remove(Val(self.0).to_uval());
                }
            }
            ConstraintResult::Ok
        }
        
        fn explain_contradictions(&self, _: &ThreeVals) -> Vec<ConstraintViolationDetail> {
            todo!()
        }
    }

    #[derive(Debug, Clone)]
    pub struct Mod(pub u8, pub u8);
    impl Stateful<u8, Val> for Mod {}
    impl Constraint<u8, ThreeVals> for Mod {
        fn check(&self, puzzle: &ThreeVals, grid: &mut DecisionGrid<u8, Val>) -> ConstraintResult<u8, Val> {
            for j in 0..3 {
                if let Some(v) = puzzle.grid.get([0, j]).map(to_value::<u8, Val>) {
                    if v.0 % self.0 != self.1 {
                        return ConstraintResult::Contradiction;
                    }
                    grid.get_mut([0, j]).0 = singleton_set(v);
                } else {
                    let s = &mut grid.get_mut([0, j]).0;
                    for v in 1..=9 {
                        if v % self.0 != self.1 {
                            s.remove(Val(v).to_uval());
                        }
                    }
                }
            }
            ConstraintResult::Ok
        }
        
        fn explain_contradictions(&self, _: &ThreeVals) -> Vec<ConstraintViolationDetail> {
            todo!()
        }
    }

    #[test]
    fn test_constraint_conjunction_simple() {
        let mut puzzle = ThreeVals { grid: UVGrid::new(ThreeVals::ROWS, ThreeVals::COLS) };
        let constraint1 = BlacklistedVal(1);
        let constraint2 = BlacklistedVal(2);
        let conjunction = ConstraintConjunction::new(constraint1, constraint2);
        let mut grid = DecisionGrid::full(ThreeVals::ROWS, ThreeVals::COLS);
        assert_eq!(conjunction.check(&puzzle, &mut grid), ConstraintResult::Ok);
        puzzle.apply([0, 0], Val(1)).unwrap();
        assert_eq!(conjunction.check(&puzzle, &mut grid), ConstraintResult::Contradiction);
        puzzle.apply([0, 0], Val(3)).unwrap();
        assert_eq!(conjunction.check(&puzzle, &mut grid), ConstraintResult::Ok);
        puzzle.apply([0, 1], Val(2)).unwrap();
        assert_eq!(conjunction.check(&puzzle, &mut grid), ConstraintResult::Contradiction);
    }

    fn unpack_set(g: &DecisionGrid<u8, Val>, index: Index) -> Vec<u8> {
        unpack_values::<u8, Val>(&g.get(index).0).iter().map(|v| v.0).collect::<Vec<u8>>()
    }

    #[test]
    fn test_constraint_conjunction_grids() {
        let puzzle = ThreeVals { grid: UVGrid::new(ThreeVals::ROWS, ThreeVals::COLS) };
        let constraint1 = Mod(2, 1);
        let constraint2 = Mod(3, 0);
        let conjunction = ConstraintConjunction::new(constraint1, constraint2);
        let mut grid = DecisionGrid::full(ThreeVals::ROWS, ThreeVals::COLS);
        assert!(conjunction.check(&puzzle, &mut grid).is_ok());
        assert_eq!(unpack_set(&grid, [0, 0]), vec![3, 9]);
        assert_eq!(unpack_set(&grid, [0, 1]), vec![3, 9]);
        assert_eq!(unpack_set(&grid, [0, 2]), vec![3, 9]);
    }
}