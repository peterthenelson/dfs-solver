use crate::core::{State, Error, Index, UInt};

/// Possible values are represented using iterators that know their size and
/// for which empty ones can be instantiated.
pub trait ActionSet<Item>: ExactSizeIterator<Item = Item> + Default {}

/// For convenience, you can use a vec::IntoIter as an ActionSet.
impl <Item> ActionSet<Item> for std::vec::IntoIter<Item> {}

/// A decision point in the puzzle. This includes the specific value that was
/// chosen, as well as the index of the cell that was modified, as well as the
/// alternative values that have not been tried yet.
#[derive(Debug, Clone)]
pub struct DecisionPoint<U: UInt, P: State<U>, A: ActionSet<P::Value>> {
    pub chosen: Option<P::Value>,
    pub index: Index,
    pub alternatives: A,
}

impl <U: UInt, P: State<U>, A: ActionSet<P::Value>> DecisionPoint<U, P, A> {
    pub fn unique(index: Index, value: P::Value) -> Self {
        DecisionPoint { chosen: Some(value), index, alternatives: A::default() }
    }

    pub fn empty() -> Self {
        DecisionPoint { chosen: None, index: [0, 0], alternatives: A::default() }
    }

    pub fn new(index: Index, alternatives: A) -> Self {
        let mut d = DecisionPoint { chosen: None, index, alternatives };
        if d.alternatives.len() > 0 {
            d.chosen = Some(d.alternatives.next().unwrap());
        }
        d
    }

    pub fn is_empty(&self) -> bool {
        self.chosen.is_none() && self.alternatives.len() == 0
    }

    pub fn advance(&mut self) -> Option<P::Value> {
        if let Some(next) = self.alternatives.next() {
            self.chosen = Some(next);
            Some(next)
        } else {
            self.chosen = None;
            None
        }
    }

    pub fn into_vec(self) -> Vec<P::Value> {
        let mut vec = Vec::new();
        if let Some(chosen) = self.chosen {
            vec.push(chosen);
        }
        for alternative in self.alternatives {
            vec.push(alternative);
        }
        vec
    }
}

/// Trait for enumerating the available actions at a particular decision point
/// (preferably in an order that leads to a faster solve). Note that the
/// posibilities must be exhaustive (e.g., if the puzzle is a sudoku, the first
/// empty cell must be one of the 9 digits--if none of them work, then the
/// puzzle has no solution). Prefer to create your DecisionPoint using either
/// the ::new, ::unique, and ::empty methods. The ::new method will take the
/// first value from the iterator and set it as the chosen value.
pub trait Strategy<U: UInt, P: State<U>> {
    type ActionSet: ActionSet<P::Value>;
    fn suggest(&self, puzzle: &P) -> Result<DecisionPoint<U, P, Self::ActionSet>, Error>;
}

/// A partial strategy that can be used to suggest actions (but requires some
/// other strategy to fall back on). This is useful for strategies that are
/// specific to a particular constraint or puzzle type, but may not be able to
/// enumerate all the possible actions.
pub trait PartialStrategy<U: UInt, P: State<U>> {
    fn suggest_partial(&self, puzzle: &P) -> Result<DecisionPoint<U, P, std::vec::IntoIter<P::Value>>, Error>;
}

/// All strategies are partial strategies.
impl <U, P, S> PartialStrategy<U, P> for S
where U: UInt, P: State<U>, S: Strategy<U, P> {
    fn suggest_partial(&self, puzzle: &P) -> Result<DecisionPoint<U, P, std::vec::IntoIter<P::Value>>, Error> {
        match self.suggest(puzzle) {
            Ok(decision) => {
                Ok(DecisionPoint::new(decision.index, decision.into_vec().into_iter()))
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

    fn suggest(&self, puzzle: &P) -> Result<DecisionPoint<U, P, Self::ActionSet>, Error> {
        for strategy in &self.partial_strategies {
            match strategy.suggest_partial(puzzle) {
                Ok(decision) => {
                    let index = decision.index;
                    let actions = decision.into_vec();
                    if !actions.is_empty() {
                        return Ok(DecisionPoint::new(index, actions.into_iter()));
                    }
                }
                Err(e) => return Err(e),
            }
        }
        match self.default_strategy.suggest(puzzle) {
            Ok(decision) => {
                let index = decision.index;
                let actions = decision.into_vec();
                return Ok(DecisionPoint::new(index, actions.into_iter()));
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;
    use crate::core::{to_value, Error, State, UVGrid, UVal, UVUnwrapped, UVWrapped, Value};

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct Val(pub u8);
    impl Value<u8> for Val {
        fn parse(_: &str) -> Result<Self, Error> { todo!() }
        fn cardinality() -> usize { 3 }
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
        fn undo(&mut self, index: Index, value: Self::Value) -> Result<(), Error> {
            self.grid.set(index, None);
            let _ = value;
            Ok(())
        }
    }

    pub struct HardcodedSuggest(pub usize, pub u8);
    impl PartialStrategy<u8, ThreeVals> for HardcodedSuggest {
        fn suggest_partial(&self, puzzle: &ThreeVals) -> Result<DecisionPoint<u8, ThreeVals, std::vec::IntoIter<Val>>, Error> {
            if puzzle.grid.get([0, self.0]).is_none() {
                Ok(DecisionPoint::new([0, self.0], vec![Val(self.1)].into_iter()))
            } else {
                Ok(DecisionPoint::new([0, 0], vec![].into_iter()))
            }
        }
    }

    pub struct FirstEmptyStrategy {}
    impl Strategy<u8, ThreeVals> for FirstEmptyStrategy {
        type ActionSet = std::vec::IntoIter<Val>;
        fn suggest(&self, puzzle: &ThreeVals) -> Result<DecisionPoint<u8, ThreeVals, Self::ActionSet>, Error> {
            for i in 0..3 {
                if puzzle.grid.get([0, i]).is_none() {
                    return Ok(DecisionPoint::new([0, i], vec![Val(1), Val(2), Val(3)].into_iter()));
                }
            }
            return Ok(DecisionPoint::new([0, 0], vec![].into_iter()));
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
            let d = strategy.suggest(&puzzle).unwrap();
            let index = d.index;
            assert_eq!(index, [0, 2]);
            assert_eq!(d.into_vec(), vec![Val(2)]);
            puzzle.apply(index, Val(2)).unwrap();
        }
        {
            let d = strategy.suggest(&puzzle).unwrap();
            let index = d.index;
            assert_eq!(index, [0, 0]);
            assert_eq!(d.into_vec(), vec![Val(1), Val(2), Val(3)]);
            puzzle.apply(index, Val(1)).unwrap();
        }
        {
            let d = strategy.suggest(&puzzle).unwrap();
            let index = d.index;
            assert_eq!(index, [0, 1]);
            assert_eq!(d.into_vec(), vec![Val(1), Val(2), Val(3)]);
        }
    }
}