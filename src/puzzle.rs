use std::borrow::Cow;
use std::fmt::Debug;
use num::traits::{PrimInt, Unsigned};

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

/// Puzzles are conceptually a grid of cells, each of which has some value drawn
/// from a finite set of possible values. (Non-rectangular grids can be
/// supported by filling in the missing cells with some sentinel value.)
pub type PuzzleIndex<const DIM: usize> = [usize; DIM];

pub trait UInt: PrimInt + Unsigned {}

impl UInt for u8 {}
impl UInt for u16 {}
impl UInt for u32 {}
impl UInt for u64 {}
impl UInt for u128 {}

/// Puzzle values are drawn from a finite set of possible values. They are
/// represented as unsigned integers, but it's entirely up to the PuzzleValue,
/// PuzzleState, and Constraint implementations to interpret them.
pub trait PuzzleValue<U: UInt>: Copy + Clone + Debug + PartialEq + Eq {
    fn cardinality() -> usize;
    fn from_uint(u: U) -> Self;
    fn to_uint(self) -> U;
}

/// Trait for representing whatever puzzle is being solved in its current state
/// of being (partially) filled in.
pub trait PuzzleState<const DIM: usize, U: UInt>: Clone + Debug {
    type Value: PuzzleValue<U>;

    fn reset(&mut self);
    fn get(&self, index: PuzzleIndex<DIM>) -> Option<Self::Value>;
    fn apply(&mut self, index: PuzzleIndex<DIM>, value: Self::Value) -> Result<(), PuzzleError>;
    fn undo(&mut self, index: PuzzleIndex<DIM>) -> Result<(), PuzzleError>;
}
