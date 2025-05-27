use std::fmt::Debug;
use crate::core::{unpack_values, CertainDecision, Decision, DecisionGrid, Error, Index, State, Stateful, UInt, Value};

#[derive(Clone, Debug, PartialEq)]
pub enum Possibilities<U: UInt, V: Value<U>> {
    Any,
    Grid(DecisionGrid<U, V>)
}

impl <U: UInt, V: Value<U>> Possibilities<U, V> {
    pub fn merge_with(&mut self, other: Possibilities<U, V>) {
        match self {
            Possibilities::Any => {
                *self = other;
            },
            Possibilities::Grid(g) => {
                if let Possibilities::Grid(g2) = other {
                    g.combine_with(&g2);
                }
            }
        }
    }
}

/// Potential violation of a constaint:
/// - Early return of a violation
/// - Early return of a certainty
/// - Remaining possibilties
pub type ConstraintResult<U, V> = Decision<U, V, Possibilities<U, V>>;

impl <U: UInt, V: Value<U>> ConstraintResult<U, V> {
    pub const fn any() -> Self {
        Decision::Other(Possibilities::Any)
    }

    pub fn grid(grid: DecisionGrid<U, V>) -> Self {
        Decision::Other(Possibilities::Grid(grid))
    }

    pub fn has_certainty<S: State<U, Value=V>>(&self, puzzle: &S) -> Option<CertainDecision<U, V>> {
        match self {
            ConstraintResult::Contradiction => None,
            ConstraintResult::Certainty(d) => Some(*d),
            Decision::Other(Possibilities::Any) => None,
            Decision::Other(Possibilities::Grid(g)) => {
                for r in 0..g.rows() {
                    for c in 0..g.cols() {
                        if puzzle.get([r, c]).is_none() {
                            let cell = &g.get([r, c]).0;
                            if cell.len() == 1 {
                                let v = unpack_values::<U, V>(cell)[0];
                                return Some(CertainDecision::new([r, c], v))
                            }
                        }
                    }
                }
                None
            }
        }
    }

    pub fn has_contradiction<S: State<U, Value=V>>(&self, puzzle: &S) -> bool {
        match self {
            ConstraintResult::Contradiction => true,
            ConstraintResult::Certainty(_) => false,
            Decision::Other(p) => {
                match p {
                    Possibilities::Any => false,
                    Possibilities::Grid(g) => {
                        for r in 0..g.rows() {
                            for c in 0..g.cols() {
                                if puzzle.get([r, c]).is_none() {
                                    let cell = &g.get([r, c]).0;
                                    if cell.is_empty() {
                                        return true;
                                    }
                                }
                            }
                        }
                        false
                    }
                }
            }
        }
    }
}

/// A full explanation of a violated constraint (for UI and debugging purposes).
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintViolationDetail {
    pub message: String,
    pub highlight: Option<Vec<Index>>,
}

/// Constraints check that the puzzle state is valid. The ideal Constraint 
/// will:
/// - Return early when it hits a contradiction or certainty.
/// - Be able to provide a useful explanation (for UI or debugging purposes) for
///   any contradictions.
/// - Be able to provide a full DecisionGrid (possibly with ranking features)
///   when neither a contradiction nor a certainty has been found (or explicitly
///   requested).
/// - Keep its internal state updated by implementing Stateful so that the grid
///   computations are not costly.
/// 
/// However, it's acceptable to use this API for other use-cases as well:
/// 1. Some constraints may not be easy to provide DecisionGrids for. E.g.,
///    maybe there's a parity constraint for the sum of a box or row or w/e.
///    In these cases, it's certainly legal to implement a constraint that only
///    ever returns a contradiction (if found) or Any (otherwise). It just makes
///    the solver's work more difficult.
/// 2. Sometimes you may have a deductive rule for determining the value of
///    cells that doesn't seem like a Constraint at all. It's find to also
///    implement these as Constraints that only ever return a certainty (if it
///    can be deduced) or Any (otherwise).
pub trait Constraint<U: UInt, S: State<U>> where Self: Stateful<U, S::Value> + Debug {
    fn check(&self, puzzle: &S, force_grid: bool) -> ConstraintResult<U, S::Value>;
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
    fn check(&self, puzzle: &S, force_grid: bool) -> ConstraintResult<U, S::Value> {
        match self.x.check(puzzle, force_grid) {
            Decision::Contradiction => Decision::Contradiction,
            Decision::Certainty(d) => Decision::Certainty(d),
            Decision::Other(mut px) => {
                match self.y.check(puzzle, force_grid) {
                    Decision::Contradiction => Decision::Contradiction,
                    Decision::Certainty(d) => Decision::Certainty(d),
                    Decision::Other(py) => {
                        px.merge_with(py);
                        Decision::Other(px)
                    }
                }
            }
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
mod tests {
    use super::*;
    use crate::core::{singleton_set, to_value, unpack_values, Error, State, Stateful, UVGrid, UVUnwrapped, UVWrapped, UVal, Value};

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct Val(pub u8);
    impl Value<u8> for Val {
        fn parse(_: &str) -> Result<Self, Error> { todo!() }
        fn cardinality() -> usize { /* actually a lie, but zero is wasted */ 10 }
        fn possiblities() -> Vec<Self> { (1..=9).map(Val).collect() }
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
        fn check(&self, puzzle: &ThreeVals, _: bool) -> ConstraintResult<u8, Val> {
            for j in 0..3 {
                if puzzle.grid.get([0, j]).map(to_value) == Some(Val(self.0)) {
                    return Decision::Contradiction;
                }
            }
            ConstraintResult::any()
        }
        
        fn explain_contradictions(&self, _: &ThreeVals) -> Vec<ConstraintViolationDetail> {
            todo!()
        }
    }

    #[derive(Debug, Clone)]
    pub struct Mod(pub u8, pub u8);
    impl Stateful<u8, Val> for Mod {}
    impl Constraint<u8, ThreeVals> for Mod {
        fn check(&self, puzzle: &ThreeVals, _: bool) -> ConstraintResult<u8, Val> {
            let mut g = DecisionGrid::new(ThreeVals::ROWS, ThreeVals::COLS);
            for j in 0..3 {
                if let Some(v) = puzzle.grid.get([0, j]).map(to_value::<u8, Val>) {
                    if v.0 % self.0 != self.1 {
                        return Decision::Contradiction;
                    }
                    g.get_mut([0, j]).0 = singleton_set(v);
                } else {
                    let s = &mut g.get_mut([0, j]).0;
                    for v in 1..=9 {
                        if v % self.0 == self.1 {
                            s.insert(Val(v).to_uval());
                        }
                    }
                }
            }
            ConstraintResult::grid(g)
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
        assert_eq!(conjunction.check(&puzzle, false), ConstraintResult::any());
        puzzle.apply([0, 0], Val(1)).unwrap();
        assert_eq!(conjunction.check(&puzzle, false), ConstraintResult::Contradiction);
        puzzle.apply([0, 0], Val(3)).unwrap();
        assert_eq!(conjunction.check(&puzzle, false), ConstraintResult::any());
        puzzle.apply([0, 1], Val(2)).unwrap();
        assert_eq!(conjunction.check(&puzzle, false), ConstraintResult::Contradiction);
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
        if let Decision::Other(Possibilities::Grid(g)) = conjunction.check(&puzzle, false) {
            assert_eq!(unpack_set(&g, [0, 0]), vec![3, 9]);
            assert_eq!(unpack_set(&g, [0, 1]), vec![3, 9]);
            assert_eq!(unpack_set(&g, [0, 2]), vec![3, 9]);
        } else {
            panic!("Expected a DecisionGrid");
        }
    }
}