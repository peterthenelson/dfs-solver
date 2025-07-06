use std::fmt::Debug;
use crate::{constraint::Constraint, core::{ConstraintResult, Index, Overlay, State, Stateful, Value}, ranker::RankingInfo};

pub type CheckFn<V, O> = fn (puzzle: &State<V, O>, ranking: &mut RankingInfo<V>) -> ConstraintResult<V>;
pub type DebugAtFn<V, O> = fn (puzzle: &State<V, O>, index: Index) -> Option<String>;
pub type DebugHighlightFn<V, O> = fn (puzzle: &State<V, O>, index: Index) -> Option<(u8, u8, u8)>;

pub struct SimpleConstraint<V: Value, O: Overlay> {
    pub name: Option<String>,
    pub check: CheckFn<V, O>,
    pub debug_str: Option<String>,
    pub debug_at: Option<DebugAtFn<V, O>>,
    pub debug_highlight: Option<DebugHighlightFn<V, O>>,
}

impl <V: Value, O: Overlay> Debug for SimpleConstraint<V, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ds) = &self.debug_str {
            write!(f, "  {}\n", ds)?;
        }
        Ok(())
    }
}

impl <V: Value, O: Overlay> Stateful<V> for SimpleConstraint<V, O> {}

impl <V: Value, O: Overlay> Constraint<V, O> for SimpleConstraint<V, O> {
    fn name(&self) -> Option<String> { self.name.clone() }

    fn check(&self, puzzle: &State<V, O>, ranking: &mut RankingInfo<V>) -> ConstraintResult<V> {
        (self.check)(puzzle, ranking)
    }

    fn debug_at(&self, puzzle: &State<V, O>, index: Index) -> Option<String> {
        self.debug_at.map(|da| da(puzzle, index)).flatten()
    }

    fn debug_highlight(&self, puzzle: &State<V, O>, index: Index) -> Option<(u8, u8, u8)> {
        self.debug_highlight.map(|dh| dh(puzzle, index)).flatten()
    }
}

#[cfg(test)]
mod test {
    use crate::{constraint::{test_util::{assert_contradiction, assert_no_contradiction}, MultiConstraint}, core::Key, ranker::StdRanker, solver::test_util::PuzzleReplay, sudoku::{four_standard_parse, FourStdOverlay, FourStdVal, StdChecker}};
    use super::*;

    // This is a 4x4 puzzle with a SimpleConstraint asserting that [0, 0] > 2.
    // Call with different givens and an expectation for it to return a
    // contradiction (or not).
    fn assert_simple_constraint_result(
        setup: &str, 
        expected: Option<&'static str>,
    ) {
        let ranker = StdRanker::default();
        let simple = SimpleConstraint {
            name: Some("[0,0] > 2".to_string()),
            check: |puzzle: &State<FourStdVal, FourStdOverlay>, _| {
                if let Some(v) = puzzle.get([0, 0]) {
                    if v.val() <= 2 {
                        return ConstraintResult::Contradiction(Key::register("CELL_0_0_NOT_OVER_2"));
                    }
                }
                ConstraintResult::Ok
            },
            debug_str: None,
            debug_at: None,
            debug_highlight: None,
        };
        let mut puzzle = four_standard_parse(setup).unwrap();
        let mut constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(puzzle.overlay()),
            simple,
        ]);
        let result = PuzzleReplay::new(&mut puzzle, &ranker, &mut constraint, None).replay().unwrap();
        if let Some(attr) = expected {
            assert_contradiction(result, attr);
        } else {
            assert_no_contradiction(result);
        }
    }

    #[test]
    fn test_simple_constraint_contradiction() {
        let setup: &str = "1 .|. .\n\
                           . .|. .\n\
                           ---+---\n\
                           . .|. .\n\
                           . .|. .\n";
        assert_simple_constraint_result(setup, Some("CELL_0_0_NOT_OVER_2"));
    }

    #[test]
    fn test_simple_constraint_valid_fill() {
        let setup: &str = "3 .|. .\n\
                           . .|. .\n\
                           ---+---\n\
                           . .|. .\n\
                           . .|. .\n";
        assert_simple_constraint_result(setup, None);
    }
}