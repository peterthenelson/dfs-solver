use std::{fmt::Debug, marker::PhantomData};
use crate::{color_util::{color_ave, color_ave2}, core::{ConstraintResult, Error, Index, Overlay, State, Stateful, Value}, ranker::RankingInfo};

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
pub trait Constraint<V: Value, O: Overlay> where Self: Stateful<V> + Debug {
    // The name of the constraint, if any. Used in UIs.
    fn name(&self) -> Option<String> { None }
    /// Check that the Constraint is satisfied by the puzzle (and any internal
    /// state from past actions). If a Constraint is able to infer useful
    /// information about what values a cell could take on, they should update
    /// the grids in the RankingInfo (in a way that further constrains them).
    fn check(&self, puzzle: &State<V, O>, ranking: &mut RankingInfo<V>) -> ConstraintResult<V>;
    /// Provide debug information at a particular grid in the puzzle (if any
    /// is available).
    fn debug_at(&self, puzzle: &State<V, O>, index: Index) -> Option<String>;
    /// Highlight the grid (again, for debug purposes).
    fn debug_highlight(&self, puzzle: &State<V, O>, index: Index) -> Option<(u8, u8, u8)>;
}

pub struct ConstraintConjunction<V, O, X, Y>
where
    V: Value, O: Overlay, X: Constraint<V, O>, Y: Constraint<V, O>
{
    pub x: X,
    pub y: Y,
    pub _marker: PhantomData<(V, O)>,
}

impl <V, O, X, Y> ConstraintConjunction<V, O, X, Y>
where
    V: Value, O: Overlay, X: Constraint<V, O>, Y: Constraint<V, O>
{
    pub fn new(x: X, y: Y) -> Self {
        ConstraintConjunction { x, y, _marker: PhantomData }
    }
}

impl <V, O, X, Y> Debug for ConstraintConjunction<V, O, X, Y>
where
    V: Value, O: Overlay, X: Constraint<V, O>, Y: Constraint<V, O>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}{:?}", self.x, self.y)
    }
}

impl <V, O, X, Y> Stateful<V> for ConstraintConjunction<V, O, X, Y>
where
    V: Value, O: Overlay, X: Constraint<V, O>, Y: Constraint<V, O>
{
    fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
    }

    fn apply(&mut self, index: Index, value: V) -> Result<(), Error> {
        let xres = self.x.apply(index, value);
        let yres = self.y.apply(index, value);
        if xres.is_err() { xres } else { yres }
    }

    fn undo(&mut self, index: Index, value: V) -> Result<(), Error> {
        let xres = self.x.undo(index, value);
        let yres = self.y.undo(index, value);
        if xres.is_err() { xres } else { yres }
    }
}

impl <V, O, X, Y> Constraint<V, O> for ConstraintConjunction<V, O, X, Y>
where
    V: Value, O: Overlay, X: Constraint<V, O>, Y: Constraint<V, O>
{
    fn check(&self, puzzle: &State<V, O>, ranking: &mut RankingInfo<V>) -> ConstraintResult<V> {
        match self.x.check(puzzle, ranking) {
            ConstraintResult::Contradiction(a) => ConstraintResult::Contradiction(a),
            ConstraintResult::Certainty(d, a) => ConstraintResult::Certainty(d, a),
            ConstraintResult::Ok => self.y.check(puzzle, ranking),
        }
    }

    fn debug_at(&self, puzzle: &State<V, O>, index: Index) -> Option<String> {
        let xd = self.x.debug_at(puzzle, index.clone());
        let yd = self.y.debug_at(puzzle, index);
        if let Some(xds) = &xd {
            if let Some(yds) = yd {
                Some(xds.clone() + "\n" + &yds)
            } else {
                xd
            }
        } else {
            yd
        }
    }

    fn debug_highlight(&self, puzzle: &State<V, O>, index: Index) -> Option<(u8, u8, u8)> {
        let xc = self.x.debug_highlight(puzzle, index.clone());
        let yc = self.y.debug_highlight(puzzle, index);
        if let Some(xrgb) = &xc {
            if let Some(yrgb) = &yc {
                Some(color_ave2(*xrgb, *yrgb))
            } else {
                xc
            }
        } else {
            yc
        }
    }
}

pub struct MultiConstraint<V: Value, O: Overlay> {
    constraints: Vec<Box<dyn Constraint<V, O>>>,
}

impl <V: Value, O: Overlay> MultiConstraint<V, O> {
    pub fn new(constraints: Vec<Box<dyn Constraint<V, O>>>) -> Self {
        MultiConstraint { constraints }
    }

    pub fn num_constraints(&self) -> usize { self.constraints.len() }

    pub fn constraint(&self, i: usize) -> &Box<dyn Constraint<V, O>> {
        &self.constraints[i]
    }
}

impl <V: Value, O: Overlay> Debug for MultiConstraint<V, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for c in &self.constraints {
            write!(f, "{:?}", c)?
        }
        Ok(())
    }
}

impl <V: Value, O: Overlay> Stateful<V> for MultiConstraint<V, O> {
    fn reset(&mut self) {
        for c in &mut self.constraints {
            c.reset();
        }
    }

    fn apply(&mut self, index: Index, value: V) -> Result<(), Error> {
        let mut res = Ok(());
        for c in &mut self.constraints {
            let maybe_err = c.apply(index, value);
            if maybe_err.is_err() {
                res = maybe_err;
            }
        }
        res
    }

    fn undo(&mut self, index: Index, value: V) -> Result<(), Error> {
        let mut res = Ok(());
        for c in &mut self.constraints {
            let maybe_err = c.undo(index, value);
            if maybe_err.is_err() {
                res = maybe_err;
            }
        }
        res
    }
}

impl <V: Value, O: Overlay> Constraint<V, O> for MultiConstraint<V, O> {
    fn name(&self) -> Option<String> { Some("MultiConstraint".to_string()) }

    fn check(&self, puzzle: &State<V, O>, ranking: &mut RankingInfo<V>) -> ConstraintResult<V> {
        for c in &self.constraints {
            match c.check(puzzle, ranking) {
                ConstraintResult::Contradiction(a) => return ConstraintResult::Contradiction(a),
                ConstraintResult::Certainty(d, a) => return ConstraintResult::Certainty(d, a),
                ConstraintResult::Ok => {},
            }
        }
        ConstraintResult::Ok
    }

    fn debug_at(&self, puzzle: &State<V, O>, index: Index) -> Option<String> {
        let somes = self.constraints.iter()
            .filter_map(|c| c.debug_at(puzzle, index.clone()))
            .collect::<Vec<String>>();
        if somes.is_empty() {
            None
        } else {
            Some(somes.join("\n"))
        }
    }

    fn debug_highlight(&self, puzzle: &State<V, O>, index: Index) -> Option<(u8, u8, u8)> {
        let somes = self.constraints.iter()
            .filter_map(|c| c.debug_highlight(puzzle, index.clone()))
            .collect::<Vec<(u8, u8, u8)>>();
        if somes.is_empty() {
            None
        } else {
            Some(color_ave(&somes))
        }
    }
}

#[cfg(any(test, feature = "test-util"))]
pub mod test_util {
    use super::*;
    use crate::core::{Value};

    pub fn assert_contradiction<V: Value>(
        cr: ConstraintResult<V>,
        expected_attribution: &'static str,
    ) {
        if let ConstraintResult::Contradiction(a) = cr {
            let actual_attribution = a.name();
            assert_eq!(
                actual_attribution, expected_attribution,
                "Expected Contradiction to be attributed to {}; got {}",
                expected_attribution, actual_attribution,
            );
        } else {
            panic!("Expected a contradiction; got: {:?}", cr);
        }
    }

    pub fn assert_no_contradiction<V: Value>(
        cr: ConstraintResult<V>,
    ) {
        if let ConstraintResult::Contradiction(a) = cr {
            panic!("Expected no contradiction; got: {:}", a.name());
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use super::test_util::*;
    use crate::core::test_util::{OneDimOverlay, TestVal};
    use crate::core::{Key, Stateful, VBitSet, VSet, VSetMut};
    use crate::ranker::{DecisionGrid, Ranker, Raw, StdRanker};

    type ThreeVals = State<TestVal, OneDimOverlay<3>>;

    #[derive(Debug, Clone)]
    pub struct BlacklistedVal(pub u8);
    impl Stateful<TestVal> for BlacklistedVal {}
    impl Constraint<TestVal, OneDimOverlay<3>> for BlacklistedVal {
        fn check(&self, puzzle: &ThreeVals, ranking: &mut RankingInfo<TestVal>) -> ConstraintResult<TestVal> {
            let grid = ranking.cells_mut();
            for j in 0..3 {
                if puzzle.get([0, j]) == Some(TestVal(self.0)) {
                    return ConstraintResult::Contradiction(Key::register("BLACKLISTED"));
                } else {
                    grid.get_mut([0, j]).0.remove(&TestVal(self.0));
                }
            }
            ConstraintResult::Ok
        }
        fn debug_at(&self, _: &ThreeVals, _: Index) -> Option<String> { None }
        fn debug_highlight(&self, _: &State<TestVal, OneDimOverlay<3>>, _: Index) -> Option<(u8, u8, u8)> { None }
    }

    #[derive(Debug, Clone)]
    pub struct Mod(pub u8, pub u8);
    impl Stateful<TestVal> for Mod {}
    impl Constraint<TestVal, OneDimOverlay<3>> for Mod {
        fn check(&self, puzzle: &ThreeVals, ranking: &mut RankingInfo<TestVal>) -> ConstraintResult<TestVal> {
            let grid = ranking.cells_mut();
            for j in 0..3 {
                if let Some(v) = puzzle.get([0, j]) {
                    if v.0 % self.0 != self.1 {
                        return ConstraintResult::Contradiction(Key::register("WRONG_MOD"));
                    }
                    *grid.get_mut([0, j]).0 = VBitSet::<TestVal>::singleton(&v);
                } else {
                    let s = &mut grid.get_mut([0, j]).0;
                    for v in 1..=9 {
                        if v % self.0 != self.1 {
                            s.remove(&TestVal(v));
                        }
                    }
                }
            }
            ConstraintResult::Ok
        }
        fn debug_at(&self, _: &ThreeVals, _: Index) -> Option<String> { None }
        fn debug_highlight(&self, _: &State<TestVal, OneDimOverlay<3>>, _: Index) -> Option<(u8, u8, u8)> { None }
    }

    #[test]
    fn test_constraint_conjunction_simple() {
        let mut puzzle = ThreeVals::new(OneDimOverlay::<3> {});
        let constraint1 = BlacklistedVal(1);
        let constraint2 = BlacklistedVal(2);
        let conjunction = ConstraintConjunction::new(constraint1, constraint2);
        let mut ranking = StdRanker::default().init_ranking(&puzzle);
        assert_eq!(conjunction.check(&puzzle, &mut ranking), ConstraintResult::Ok);
        puzzle.apply([0, 0], TestVal(1)).unwrap();
        assert_contradiction(conjunction.check(&puzzle, &mut ranking), "BLACKLISTED");
        puzzle.undo([0, 0], TestVal(1)).unwrap();
        puzzle.apply([0, 0], TestVal(3)).unwrap();
        assert_eq!(conjunction.check(&puzzle, &mut ranking), ConstraintResult::Ok);
        puzzle.apply([0, 1], TestVal(2)).unwrap();
        assert_contradiction(conjunction.check(&puzzle, &mut ranking), "BLACKLISTED");
    }

    #[test]
    fn test_multi_constraint_simple() {
        let mut puzzle = ThreeVals::new(OneDimOverlay::<3> {});
        let constraint = MultiConstraint::new(vec_box::vec_box![
            BlacklistedVal(1), BlacklistedVal(2),
        ]);
        let mut ranking = StdRanker::default().init_ranking(&puzzle);
        assert_eq!(constraint.check(&puzzle, &mut ranking), ConstraintResult::Ok);
        puzzle.apply([0, 0], TestVal(1)).unwrap();
        assert_contradiction(constraint.check(&puzzle, &mut ranking), "BLACKLISTED");
        puzzle.undo([0, 0], TestVal(1)).unwrap();
        puzzle.apply([0, 0], TestVal(3)).unwrap();
        assert_eq!(constraint.check(&puzzle, &mut ranking), ConstraintResult::Ok);
        puzzle.apply([0, 1], TestVal(2)).unwrap();
        assert_contradiction(constraint.check(&puzzle, &mut ranking), "BLACKLISTED");
    }

    fn unpack_set(g: &DecisionGrid<TestVal, Raw>, index: Index) -> Vec<u8> {
        g.get(index).0.iter().map(|v| v.0).collect::<Vec<u8>>()
    }

    #[test]
    fn test_constraint_conjunction_grids() {
        let puzzle = ThreeVals::new(OneDimOverlay::<3> {});
        let constraint1 = Mod(2, 1);
        let constraint2 = Mod(3, 0);
        let conjunction = ConstraintConjunction::new(constraint1, constraint2);
        let mut ranking = StdRanker::default().init_ranking(&puzzle);
        match conjunction.check(&puzzle, &mut ranking) {
            ConstraintResult::Contradiction(a) => panic!("Unexpected contradiction: {}", a.name()),
            _ => {},
        };
        assert_eq!(unpack_set(ranking.cells(), [0, 0]), vec![3, 9]);
        assert_eq!(unpack_set(ranking.cells(), [0, 1]), vec![3, 9]);
        assert_eq!(unpack_set(ranking.cells(), [0, 2]), vec![3, 9]);
    }

    #[test]
    fn test_multi_constraint_grids() {
        let puzzle = ThreeVals::new(OneDimOverlay::<3> {});
        let constraint = MultiConstraint::new(vec_box::vec_box![
            Mod(2, 1), Mod(3, 0)
        ]);
        let mut ranking = StdRanker::default().init_ranking(&puzzle);
        match constraint.check(&puzzle, &mut ranking) {
            ConstraintResult::Contradiction(a) => panic!("Unexpected contradiction: {}", a.name()),
            _ => {},
        };
        assert_eq!(unpack_set(ranking.cells(), [0, 0]), vec![3, 9]);
        assert_eq!(unpack_set(ranking.cells(), [0, 1]), vec![3, 9]);
        assert_eq!(unpack_set(ranking.cells(), [0, 2]), vec![3, 9]);
    }
}