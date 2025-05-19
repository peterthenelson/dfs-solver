use crate::core::{State, Error, Index, UInt};

/// Trait for enumerating the available actions at a particular decision point
/// (preferably in an order that leads to a faster solve). Note that the
/// posibilities must be exhaustive (e.g., if the puzzle is a sudoku, the first
/// empty cell must be one of the 9 digits--if none of them work, then the
/// puzzle has no solution).
pub trait Strategy<U: UInt, P: State<U>> {
    type ActionSet: Iterator<Item = P::Value>;
    fn suggest(&self, puzzle: &P) -> Result<(Index, Self::ActionSet), Error>;
    fn empty_action_set() -> Self::ActionSet;
}

/// A partial strategy that can be used to suggest actions (but requires some
/// other strategy to fall back on). This is useful for strategies that are
/// specific to a particular constraint or puzzle type, but may not be able to
/// enumerate all the possible actions.
pub trait PartialStrategy<U: UInt, P: State<U>> {
    fn suggest_partial(&self, puzzle: &P) -> Result<(Index, Vec<P::Value>), Error>;
}

/// All strategies are partial strategies.
impl <U, P, S> PartialStrategy<U, P> for S
where U: UInt, P: State<U>, S: Strategy<U, P> {
    fn suggest_partial(&self, puzzle: &P) -> Result<(Index, Vec<P::Value>), Error> {
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
pub struct CompositeStrategy<'a, U, P, S>
where U: UInt, P: State<U>, S: Strategy<U, P> {
    default_strategy: S,
    partial_strategies: Vec<&'a dyn PartialStrategy<U, P>>,
    p_u: std::marker::PhantomData<U>,
    p_p: std::marker::PhantomData<P>,
}

impl <'a, U, P, S> CompositeStrategy<'a, U, P, S>
where U: UInt, P: State<U>, S: Strategy<U, P> {
    pub fn new(default_strategy: S, partial_strategies: Vec<&'a dyn PartialStrategy<U, P>>) -> Self {
        CompositeStrategy {
            default_strategy,
            partial_strategies,
            p_u: std::marker::PhantomData,
            p_p: std::marker::PhantomData,
        }
    }
}
impl <'a, U, P, S> Strategy<U, P>
for CompositeStrategy<'a, U, P, S>
where U: UInt, P: State<U>, S: Strategy<U, P> {
    type ActionSet = std::vec::IntoIter<P::Value>;

    fn suggest(&self, puzzle: &P) -> Result<(Index, Self::ActionSet), Error> {
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
    use super::*;
    use std::vec;
    use crate::core::{to_value, Error, State, UVGrid, UVal, UValUnwrapped, UValWrapped, Value};

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct Val(pub u8);
    impl Value<u8> for Val {
        fn parse(_: &str) -> Result<Self, Error> { todo!() }
        fn cardinality() -> usize { 3 }
        fn from_uval(u: UVal<u8, UValUnwrapped>) -> Self { Val(u.value()) }
        fn to_uval(self) -> UVal<u8, UValWrapped> { UVal::new(self.0) }
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
        fn undo(&mut self, index: Index) -> Result<(), Error> {
            self.grid.set(index, None);
            Ok(())
        }
    }

    pub struct HardcodedSuggest(pub usize, pub u8);
    impl PartialStrategy<u8, ThreeVals> for HardcodedSuggest {
        fn suggest_partial(&self, puzzle: &ThreeVals) -> Result<(Index, Vec<Val>), Error> {
            if puzzle.grid.get([0, self.0]).is_none() {
                Ok(([0, self.0], vec![Val(self.1)]))
            } else {
                Ok(([0, 0], vec![]))
            }
        }
    }

    pub struct FirstEmptyStrategy {}
    impl Strategy<u8, ThreeVals> for FirstEmptyStrategy {
        type ActionSet = std::vec::IntoIter<Val>;
        fn suggest(&self, puzzle: &ThreeVals) -> Result<(Index, Self::ActionSet), Error> {
            for i in 0..3 {
                if puzzle.grid.get([0, i]).is_none() {
                    return Ok(([0, i], vec![Val(1), Val(2), Val(3)].into_iter()));
                }
            }
            return Ok(([0, 0], vec![].into_iter()));
        }
        fn empty_action_set() -> Self::ActionSet {
            vec![].into_iter()
        }
    }

    #[test]
    fn test_composite_strategy() {
        let mut puzzle = ThreeVals { grid: UVGrid::new(ThreeVals::ROWS, ThreeVals::COLS) };
        let partial = HardcodedSuggest(2, 2);
        let strategy= CompositeStrategy::new(
            FirstEmptyStrategy {},
            vec![&partial],
        );
        {
            let (index, action_set) = strategy.suggest(&puzzle).unwrap();
            assert_eq!(index, [0, 2]);
            assert_eq!(action_set.collect::<Vec<_>>(), vec![Val(2)]);
            puzzle.apply(index, Val(2)).unwrap();
        }
        {
            let (index, action_set) = strategy.suggest(&puzzle).unwrap();
            assert_eq!(index, [0, 0]);
            assert_eq!(action_set.collect::<Vec<_>>(), vec![Val(1), Val(2), Val(3)]);
            puzzle.apply(index, Val(1)).unwrap();
        }
        {
            let (index, action_set) = strategy.suggest(&puzzle).unwrap();
            assert_eq!(index, [0, 1]);
            assert_eq!(action_set.collect::<Vec<_>>(), vec![Val(1), Val(2), Val(3)]);
        }
    }
}