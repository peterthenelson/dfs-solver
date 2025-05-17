use crate::puzzle::{PuzzleState, PuzzleError, PuzzleIndex, UInt};

/// Trait for enumerating the available actions at a particular decision point
/// (preferably in an order that leads to a faster solve). Note that the
/// posibilities must be exhaustive (e.g., if the puzzle is a sudoku, the first
/// empty cell must be one of the 9 digits--if none of them work, then the
/// puzzle has no solution).
pub trait Strategy<const DIM: usize, U: UInt, P: PuzzleState<DIM, U>> {
    type ActionSet: Iterator<Item = P::Value>;
    fn suggest(&self, puzzle: &P) -> Result<(PuzzleIndex<DIM>, Self::ActionSet), PuzzleError>;
    fn empty_action_set() -> Self::ActionSet;
}

/// A partial strategy that can be used to suggest actions (but requires some
/// other strategy to fall back on). This is useful for strategies that are
/// specific to a particular constraint or puzzle type, but may not be able to
/// enumerate all the possible actions.
pub trait PartialStrategy<const DIM: usize, U: UInt, P: PuzzleState<DIM, U>> {
    fn suggest_partial(&self, puzzle: &P) -> Result<(PuzzleIndex<DIM>, Vec<P::Value>), PuzzleError>;
}

/// All strategies are partial strategies.
impl <const DIM: usize, U, P, S> PartialStrategy<DIM, U, P> for S
where U: UInt, P: PuzzleState<DIM, U>, S: Strategy<DIM, U, P> {
    fn suggest_partial(&self, puzzle: &P) -> Result<(PuzzleIndex<DIM>, Vec<P::Value>), PuzzleError> {
        match self.suggest(puzzle) {
            Ok((index, action_set)) => {
                let mut actions = Vec::new();
                for action in action_set {
                    actions.push(action);
                }
                Ok((index, actions))
            }
            Err(e) => Err(e),
        }
    }
}

/// A composite strategy that combines multiple strategies into one.
pub struct CompositeStrategy<'a, const DIM: usize, U, P, S>
where U: UInt, P: PuzzleState<DIM, U>, S: Strategy<DIM, U, P> {
    default_strategy: S,
    partial_strategies: Vec<&'a dyn PartialStrategy<DIM, U, P>>,
    p_u: std::marker::PhantomData<U>,
    p_p: std::marker::PhantomData<P>,
}

impl <'a, const DIM: usize, U, P, S> CompositeStrategy<'a, DIM, U, P, S>
where U: UInt, P: PuzzleState<DIM, U>, S: Strategy<DIM, U, P> {
    pub fn new(default_strategy: S, partial_strategies: Vec<&'a dyn PartialStrategy<DIM, U, P>>) -> Self {
        CompositeStrategy {
            default_strategy,
            partial_strategies,
            p_u: std::marker::PhantomData,
            p_p: std::marker::PhantomData,
        }
    }
}
impl <'a, const DIM: usize, U, P, S> Strategy<DIM, U, P>
for CompositeStrategy<'a, DIM, U, P, S>
where U: UInt, P: PuzzleState<DIM, U>, S: Strategy<DIM, U, P> {
    type ActionSet = std::vec::IntoIter<P::Value>;

    fn suggest(&self, puzzle: &P) -> Result<(PuzzleIndex<DIM>, Self::ActionSet), PuzzleError> {
        for strategy in &self.partial_strategies {
            match strategy.suggest_partial(puzzle) {
                Ok((index, action_set)) => {
                    let mut actions = Vec::new();
                    for action in action_set {
                        actions.push(action);
                    }
                    if !actions.is_empty() {
                        return Ok((index, actions.into_iter()));
                    }
                }
                Err(e) => return Err(e),
            }
        }
        match self.default_strategy.suggest(puzzle) {
            Ok((index, action_set)) => {
                let mut actions = Vec::new();
                for action in action_set {
                    actions.push(action);
                }
                return Ok((index, actions.into_iter()));
            }
            Err(e) => return Err(e),
        }
    }

    fn empty_action_set() -> Self::ActionSet {
        vec![].into_iter()
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use crate::puzzle::{PuzzleError, PuzzleState, PuzzleValue};

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct Val(pub u8);
    impl PuzzleValue<u8> for Val {
        fn cardinality() -> usize { 3 }
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

    pub struct HardcodedSuggest(pub usize, pub u8);
    impl PartialStrategy<1, u8, ThreeVals> for HardcodedSuggest {
        fn suggest_partial(&self, puzzle: &ThreeVals) -> Result<(PuzzleIndex<1>, Vec<Val>), PuzzleError> {
            if puzzle.grid[self.0].is_none() {
                Ok(([self.0], vec![Val(self.1)]))
            } else {
                Ok(([0], vec![]))
            }
        }
    }

    pub struct FirstEmptyStrategy {}
    impl Strategy<1, u8, ThreeVals> for FirstEmptyStrategy {
        type ActionSet = std::vec::IntoIter<Val>;
        fn suggest(&self, puzzle: &ThreeVals) -> Result<(PuzzleIndex<1>, Self::ActionSet), PuzzleError> {
            for i in 0..3 {
                if puzzle.grid[i].is_none() {
                    return Ok(([i], vec![Val(1), Val(2), Val(3)].into_iter()));
                }
            }
            return Ok(([0], vec![].into_iter()));
        }
        fn empty_action_set() -> Self::ActionSet {
            vec![].into_iter()
        }
    }

    #[test]
    fn test_composite_strategy() {
        let mut puzzle = ThreeVals { grid: [None, None, None] };
        let partial = HardcodedSuggest(2, 2);
        let strategy= CompositeStrategy::new(
            FirstEmptyStrategy {},
            vec![&partial],
        );
        {
            let (index, action_set) = strategy.suggest(&puzzle).unwrap();
            assert_eq!(index, [2]);
            assert_eq!(action_set.collect::<Vec<_>>(), vec![Val(2)]);
            puzzle.apply(index, Val(2)).unwrap();
        }
        {
            let (index, action_set) = strategy.suggest(&puzzle).unwrap();
            assert_eq!(index, [0]);
            assert_eq!(action_set.collect::<Vec<_>>(), vec![Val(1), Val(2), Val(3)]);
            puzzle.apply(index, Val(1)).unwrap();
        }
        {
            let (index, action_set) = strategy.suggest(&puzzle).unwrap();
            assert_eq!(index, [1]);
            assert_eq!(action_set.collect::<Vec<_>>(), vec![Val(1), Val(2), Val(3)]);
        }
    }
}