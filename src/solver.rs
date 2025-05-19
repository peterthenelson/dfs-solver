use std::fmt::Debug;
use crate::core::{Error, Index, State, UInt};
use crate::constraint::{Constraint, ConstraintResult};
use crate::strategy::{DecisionPoint, Strategy};

/// The state of the DFS solver. At any point in time, the solver is either
/// advancing (ready to take a new action), backtracking (undoing actions),
/// solved (has found a solution), or exhausted (no more actions to take).
#[derive(Debug, PartialEq, Clone, Copy, Eq)]
pub enum DfsSolverState {
    Advancing,
    Backtracking,
    Solved,
    Exhausted,
}

// A view on the state and associated data for the solver.
pub trait DfsSolverView<U: UInt, P: State<U>> {
    fn get_state(&self) -> DfsSolverState;
    fn is_done(&self) -> bool;
    fn is_valid(&self) -> bool;
    fn get_violations(&self) -> ConstraintResult;
    fn get_puzzle(&self) -> &P;
}

/// DFS solver. If you want a lower-level API that allows for more control over
/// the solving process, you can directly use this. Most users should prefer
/// FindFirstSolution or FindAllSolutions, which are higher-level APIs. However,
/// if you are implementing a UI or debugging, this API may be useful.
pub struct DfsSolver<'a, U, P, S, C>
where U: UInt, P: State<U>, S: Strategy<U, P>, C: Constraint<U, P> {
    puzzle: &'a mut P,
    strategy: &'a S,
    constraint: &'a C,
    violation: ConstraintResult,
    stack: Vec<DecisionPoint<U, P, S::ActionSet>>,
    state: DfsSolverState,
}

impl <'a, U, P, S, C> DfsSolverView<U, P>
for DfsSolver<'a, U, P, S, C>
where U: UInt, P: State<U>, S: Strategy<U, P>, C: Constraint<U, P> {
    fn get_state(&self) -> DfsSolverState {
        self.state
    }

    fn is_done(&self) -> bool {
        self.state == DfsSolverState::Solved || self.state == DfsSolverState::Exhausted
    }

    fn is_valid(&self) -> bool {
        self.violation.is_none()
    }

    fn get_violations(&self) -> ConstraintResult {
        self.violation.clone()
    }

    fn get_puzzle(&self) -> &P {
        self.puzzle
    }
}

const PUZZLE_ALREADY_DONE: Error = Error::new_const("Puzzle already done");
const NO_CHOICE: Error = Error::new_const("Decision point has no choice");
 
impl <'a, U, P, S, C> DfsSolver<'a, U, P, S, C>
where U: UInt, P: State<U>, S: Strategy<U, P>, C: Constraint<U, P> {
    pub fn new(
        puzzle: &'a mut P,
        strategy: &'a S,
        constraint: &'a C, 
    ) -> Self {
        DfsSolver {
            puzzle,
            strategy,
            constraint,
            violation: ConstraintResult::NoViolation,
            stack: Vec::new(),
            state: DfsSolverState::Advancing,
        }
    }

    fn apply(&mut self, decision: DecisionPoint<U, P, S::ActionSet>, details: bool) -> Result<(), Error> {
        if self.is_done() {
            return Err(PUZZLE_ALREADY_DONE);
        } else if decision.chosen.is_none() {
            return Err(NO_CHOICE);
        }
        self.puzzle.apply(decision.index, decision.chosen.unwrap())?;
        self.stack.push(decision);
        self.violation = self.constraint.check(self.puzzle, details);
        self.state = if self.violation.is_none() {
            DfsSolverState::Advancing
        } else {
            DfsSolverState::Backtracking
        };
        return Ok(());
    }

    pub fn manual_step(&mut self, index: Index, value: P::Value, details: bool) -> Result<(), Error> {
        self.apply(DecisionPoint {
            chosen: Some(value),
            index,
            alternatives: S::ActionSet::default(),
         }, details)
    }

    pub fn force_backtrack(&mut self) -> bool {
        if self.state == DfsSolverState::Exhausted {
            return false;
        }
        self.state = DfsSolverState::Backtracking;
        true
    }

    pub fn step(&mut self, details: bool) -> Result<(), Error> {
        match self.state {
            DfsSolverState::Solved => Err(PUZZLE_ALREADY_DONE),
            DfsSolverState::Exhausted => Err(PUZZLE_ALREADY_DONE),
            DfsSolverState::Advancing => {
                // Take a new action
                let decision = self.strategy.suggest(self.puzzle)?;
                if decision.chosen.is_some() {
                    self.apply(decision, details)?;
                } else {
                    self.state = DfsSolverState::Solved;
                }
                Ok(())
            }
            DfsSolverState::Backtracking => {
                if self.stack.is_empty() {
                    self.state = DfsSolverState::Exhausted;
                    return Ok(());
                }
                // Backtrack, attempting to advance an existing action set
                let mut decision = self.stack.pop().unwrap();
                self.puzzle.undo(decision.index, decision.chosen.unwrap())?;
                match decision.advance() {
                    Some(_) => {
                        self.apply(decision, details)?;
                        Ok(())
                    }
                    None => Ok(()),
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.puzzle.reset();
        self.violation = ConstraintResult::NoViolation;
        self.stack.clear();
        self.state = DfsSolverState::Advancing;
    }
}

/// Find first solution to the puzzle using the given strategy and constraints.
pub struct FindFirstSolution<'a, U, P, S, C>(DfsSolver<'a, U, P, S, C>, bool)
where U: UInt, P: State<U>, S: Strategy<U, P>, C: Constraint<U, P>;

impl <'a, U, P, S, C> DfsSolverView<U, P>
for FindFirstSolution<'a, U, P, S, C>
where U: UInt, P: State<U>, S: Strategy<U, P>, C: Constraint<U, P> {
    fn get_state(&self) -> DfsSolverState { self.0.get_state() }
    fn is_done(&self) -> bool { self.0.is_done() }
    fn is_valid(&self) -> bool { self.0.is_valid() }
    fn get_violations(&self) -> ConstraintResult { self.0.get_violations() }
    fn get_puzzle(&self) -> &P { self.0.get_puzzle() }
}

impl <'a, U, P, S, C> FindFirstSolution<'a, U, P, S, C>
where U: UInt, P: State<U>, S: Strategy<U, P>, C: Constraint<U, P> {
    pub fn new(
        puzzle: &'a mut P,
        strategy: &'a S,
        constraint: &'a C,
        details: bool,
    ) -> Self {
        FindFirstSolution(DfsSolver::new(puzzle, strategy, constraint), details)
    }

    pub fn step(&mut self) -> Result<&dyn DfsSolverView<U, P>, Error> {
        self.0.step(self.1)?;
        Ok(&self.0)
    }

    pub fn solve(&mut self) -> Result<Option<&dyn DfsSolverView<U, P>>, Error> {
        while !self.0.is_done() {
            self.step()?;
        }
        if self.0.is_valid() {
            return Ok(Some(&self.0));
        } else {
            return Ok(None);
        }
    }
}

/// Find all solutions to the puzzle using the given strategy and constraints.
pub struct FindAllSolutions<'a, U, P, S, C>(DfsSolver<'a, U, P, S, C>, bool)
where U: UInt, P: State<U>, S: Strategy<U, P>, C: Constraint<U, P>;

impl <'a, U, P, S, C> DfsSolverView<U, P>
for FindAllSolutions<'a, U, P, S, C>
where U: UInt, P: State<U>, S: Strategy<U, P>, C: Constraint<U, P> {
    fn get_state(&self) -> DfsSolverState { self.0.get_state() }
    fn is_done(&self) -> bool { self.0.get_state() == DfsSolverState::Exhausted }
    fn is_valid(&self) -> bool { self.0.is_valid() }
    fn get_violations(&self) -> ConstraintResult { self.0.get_violations() }
    fn get_puzzle(&self) -> &P { self.0.get_puzzle() }
}

impl <'a, U, P, S, C> FindAllSolutions<'a, U, P, S, C>
where U: UInt, P: State<U>, S: Strategy<U, P>, C: Constraint<U, P> {
    pub fn new(
        puzzle: &'a mut P,
        strategy: &'a S,
        constraint: &'a C,
        details: bool,
    ) -> Self {
        FindAllSolutions(DfsSolver::new(puzzle, strategy, constraint), details)
    }

    pub fn step(&mut self) -> Result<&dyn DfsSolverView<U, P>, Error> {
        if self.0.state == DfsSolverState::Solved {
            self.0.force_backtrack();
        }
        self.0.step(self.1)?;
        Ok(&self.0)
    }
}

#[cfg(test)]
mod test {
    use crate::constraint::ConstraintViolationDetail;
    use crate::core::{to_value, UVGrid, UVal, UValUnwrapped, UValWrapped, Value};
    use crate::strategy::{CompositeStrategy, PartialStrategy};
    use super::*;

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct GwValue(pub u8);
    impl Value<u8> for GwValue {
        fn parse(_: &str) -> Result<Self, Error> { todo!() }
        fn cardinality() -> usize { 10 }
        fn from_uval(u: UVal<u8, UValUnwrapped>) -> Self { GwValue(u.value()) }
        fn to_uval(self) -> UVal<u8, UValWrapped> { UVal::new(self.0) }
    }

    #[derive(Clone, Debug)]
    struct GwLine {
        digits: UVGrid<u8>,
    }

    impl GwLine {
        pub fn new() -> Self {
            GwLine {
                digits: UVGrid::<u8>::new(Self::ROWS, Self::COLS),
            }
        }
    }

    impl std::fmt::Display for GwLine {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for i in 0..8 {
                if let Some(v) = self.digits.get([0, i]) {
                    write!(f, "{}", to_value::<u8, GwValue>(v).0)?;
                } else {
                    write!(f, ".")?;
                }
            }
            Ok(())
        }
    }

    impl State<u8> for GwLine {
        type Value = GwValue;
        const ROWS: usize = 1;
        const COLS: usize = 8;

        fn reset(&mut self) {
            self.digits = UVGrid::<u8>::new(Self::ROWS, Self::COLS);
        }

        fn get(&self, index: Index) -> Option<Self::Value> {
            if index[0] >= 1 || index[1] >= 8 {
                return None;
            }
            self.digits.get(index).map(to_value)
        }

        fn apply(&mut self, index: Index, value: Self::Value) -> Result<(), Error> {
            if index[0] >= 1 || index[1] >= 8 {
                return Err(Error::new_const("Index out of bounds"));
            }
            if self.digits.get(index).is_some() {
                return Err(Error::new_const("Cell already filled"));
            }
            self.digits.set(index, Some(value.to_uval()));
            Ok(())
        }

        fn undo(&mut self, index: Index, value: GwValue) -> Result<(), Error> {
            if index[0] >=1 || index[1] >= 8 {
                return Err(Error::new_const("Index out of bounds"));
            } else if self.digits.get(index).is_none() {
                return Err(Error::new_const("No such action to undo"));
            } else if self.digits.get(index).unwrap() != value.to_uval() {
                return Err(Error::new_const("Value does not match"));
            }
            self.digits.set(index, None);
            Ok(())
        }
    }

    struct GwLineConstraint {}
    impl Constraint<u8, GwLine> for GwLineConstraint {
        fn check(&self, puzzle: &GwLine, details: bool) -> ConstraintResult {
            let mut violations = Vec::new();
            for i in 0..8 {
                if puzzle.digits.get([0, i]).is_none() {
                    continue;
                }
                let i_val = to_value::<u8, GwValue>(puzzle.digits.get([0, i]).unwrap()).0;
                for j in i+1..8 {
                    if puzzle.digits.get([0, j]).is_none() {
                        continue;
                    }
                    let j_val = to_value::<u8, GwValue>(puzzle.digits.get([0, j]).unwrap()).0;
                    if i_val == j_val {
                        if details {
                            violations.push(ConstraintViolationDetail {
                                message: format!("Digits with duplicate value: [{}] == [{}] == {}", i, j, i_val),
                                highlight: Some(vec![[0, i], [0, j]]),
                            });
                        } else {
                            return ConstraintResult::Simple("Duplicate digits");
                        }
                    }
                    let diff: i16 = (i_val as i16) - (j_val as i16);
                    if j == i+1 && diff.abs() < 5 {
                        if details {
                            violations.push(ConstraintViolationDetail {
                                message: format!("Adjacent digits too close: [{}]={} and [{}]={}", i, i_val, j, j_val),
                                highlight: Some(vec![[0, i], [0, j]]),
                            });
                        } else {
                            return ConstraintResult::Simple("Adjacent digits too close");
                        }
                    }
                }
            }
            if violations.len() > 0 {
                return ConstraintResult::Details(violations);
            } else {
                ConstraintResult::NoViolation
            }
        }
    }

    struct GwLineStrategy {}
    impl Strategy<u8, GwLine> for GwLineStrategy {
        type ActionSet = std::vec::IntoIter<GwValue>;

        fn suggest(&self, puzzle: &GwLine) -> Result<DecisionPoint<u8, GwLine, Self::ActionSet>, Error> {
            for i in 0..8 {
                if puzzle.digits.get([0, i]).is_none() {
                    return Ok(DecisionPoint::new(
                        [0, i],
                        vec![1, 2, 3, 4, 5, 6, 7, 8, 9].into_iter().map(GwValue).collect::<Vec<_>>().into_iter())
                    );
                }
            }
            Ok(DecisionPoint::empty())
        }
    }

    struct GwLineStrategyPartial {}
    impl PartialStrategy<u8, GwLine> for GwLineStrategyPartial {
        fn suggest_partial(&self, puzzle: &GwLine) -> Result<DecisionPoint<u8, GwLine, std::vec::IntoIter<GwValue>>, Error> {
            // This is a partial strategy that only affects the first digit and
            // avoids guessing things that can't work.
            if puzzle.digits.get([0, 0]).is_none() {
                return Ok(DecisionPoint::new(
                    [0, 0],
                    vec![4, 6].into_iter().map(GwValue).collect::<Vec<_>>().into_iter())
                );
            }
            Ok(DecisionPoint::empty())
        }
    }

    #[test]
    fn german_whispers_constraint() {
        let mut puzzle = GwLine::new();
        let constraint = GwLineConstraint {};
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::NoViolation);
        puzzle.apply([0, 0], GwValue(1)).unwrap();
        puzzle.apply([0, 3], GwValue(2)).unwrap();
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::NoViolation);
        puzzle.apply([0, 5], GwValue(1)).unwrap();
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::Simple("Duplicate digits"));
        puzzle.undo([0, 5], GwValue(1)).unwrap();
        puzzle.apply([0, 1], GwValue(3)).unwrap();
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::Simple("Adjacent digits too close"));
        puzzle.undo([0, 1], GwValue(3)).unwrap();
        puzzle.apply([0, 1], GwValue(6)).unwrap();
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::NoViolation);
    }

    #[test]
    fn german_whispers_find() -> Result<(), Error> {
        let mut puzzle = GwLine::new();
        let strategy = GwLineStrategy {};
        let constraint = GwLineConstraint {};
        let mut finder = FindFirstSolution::new(&mut puzzle, &strategy, &constraint, false);
        let maybe_solution = finder.solve()?;
        assert!(maybe_solution.is_some());
        assert_eq!(maybe_solution.unwrap().get_puzzle().to_string(), "49382716");
        Ok(())
    }

    #[test]
    fn german_whispers_trace() -> Result<(), Error> {
        let mut puzzle = GwLine::new();
        let strategy = GwLineStrategy {};
        let constraint = GwLineConstraint {};
        let mut finder = FindFirstSolution::new(&mut puzzle, &strategy, &constraint, true);
        let mut steps: usize = 0;
        let mut violation_count: usize = 0;
        while !finder.is_done() {
            finder.step()?;
            steps += 1;
            violation_count += match finder.get_violations() {
                ConstraintResult::Simple(_) => 1,
                ConstraintResult::Details(details) => {
                    details.len()
                },
                ConstraintResult::NoViolation => 0,
            }
        }
        assert!(finder.is_valid());
        assert!(steps > 100);
        assert!(violation_count > 100);
        Ok(())
    }

    #[test]
    fn german_whispers_all() -> Result<(), Error> {
        let mut puzzle = GwLine::new();
        let strategy = GwLineStrategy {};
        let constraint = GwLineConstraint {};
        let mut finder = FindAllSolutions::new(&mut puzzle, &strategy, &constraint, false);
        let mut steps: usize = 0;
        let mut solution_count: usize = 0;
        while !finder.is_done() {
            finder.step()?;
            steps += 1;
            solution_count += if finder.get_state() == DfsSolverState::Solved { 1 } else { 0 };
        }
        assert!(steps > 2500);
        assert_eq!(solution_count, 2);
        Ok(())
    }

    #[test]
    fn german_whispers_all_fast() -> Result<(), Error> {
        let mut puzzle = GwLine::new();
        let partial = GwLineStrategyPartial {};
        let strategy = CompositeStrategy::new(GwLineStrategy {}, vec![&partial]);
        let constraint = GwLineConstraint {};
        let mut finder = FindAllSolutions::new(&mut puzzle, &strategy, &constraint, false);
        let mut steps: usize = 0;
        let mut solution_count: usize = 0;
        while !finder.is_done() {
            finder.step()?;
            steps += 1;
            solution_count += if finder.get_state() == DfsSolverState::Solved { 1 } else { 0 };
        }
        assert!(steps < 1000);
        assert_eq!(solution_count, 2);
        Ok(())
    }
}