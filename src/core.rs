use std::borrow::Cow;
use std::fmt::Debug;

/// Error type for the solver. This is used to indicate something wrong with
/// either the puzzle/strategy/constraints or with the algorithm itself.
/// Violations of constraints or exhaustion of the search space are not errors.
#[derive(Debug, Clone, PartialEq)]
pub struct PuzzleError(Cow<'static, str>);
impl PuzzleError {
    pub const fn new_const(s: &'static str) -> Self {
        PuzzleError(Cow::Borrowed(s))
    }

    pub fn new<S: Into<String>>(s: S) -> Self {
        PuzzleError(Cow::Owned(s.into()))
    }
}

/// Trait for representing whatever puzzle is being solved.
pub trait PuzzleState<A>: Clone + Debug {
    fn reset(&mut self);
    fn apply(&mut self, action: &A) -> Result<(), PuzzleError>;
    fn undo(&mut self, action: &A) -> Result<(), PuzzleError>;
}
