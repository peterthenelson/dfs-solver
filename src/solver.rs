use std::fmt::Debug;
use crate::puzzle::{PuzzleError, PuzzleIndex, PuzzleState, UInt};
use crate::constraint::{Constraint, ConstraintResult};
use crate::strategy::Strategy;

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
pub trait DfsSolverView<const DIM: usize, U: UInt, P: PuzzleState<DIM, U>> {
    fn get_state(&self) -> DfsSolverState;
    fn is_done(&self) -> bool;
    fn is_valid(&self) -> bool;
    fn get_violations(&self) -> ConstraintResult<DIM>;
    fn get_puzzle(&self) -> &P;
}

/// DFS solver. If you want a lower-level API that allows for more control over
/// the solving process, you can directly use this. Most users should prefer
/// FindFirstSolution or FindAllSolutions, which are higher-level APIs. However,
/// if you are implementing a UI or debugging, this API may be useful.
pub struct DfsSolver<'a, const DIM: usize, U, P, S, C>
where U: UInt, P: PuzzleState<DIM, U>, S: Strategy<DIM, U, P>, C: Constraint<DIM, U, P> {
    puzzle: &'a mut P,
    strategy: &'a S,
    constraint: &'a C,
    violation: ConstraintResult<DIM>,
    stack: Vec<(PuzzleIndex<DIM>, <S as Strategy<DIM, U, P>>::ActionSet)>,
    state: DfsSolverState,
}

impl <'a, const DIM: usize, U, P, S, C> DfsSolverView<DIM, U, P>
for DfsSolver<'a, DIM, U, P, S, C>
where U: UInt, P: PuzzleState<DIM, U>, S: Strategy<DIM, U, P>, C: Constraint<DIM, U, P> {
    fn get_state(&self) -> DfsSolverState {
        self.state
    }

    fn is_done(&self) -> bool {
        self.state == DfsSolverState::Solved || self.state == DfsSolverState::Exhausted
    }

    fn is_valid(&self) -> bool {
        self.violation.is_none()
    }

    fn get_violations(&self) -> ConstraintResult<DIM> {
        self.violation.clone()
    }

    fn get_puzzle(&self) -> &P {
        self.puzzle
    }
}

const PUZZLE_ALREADY_DONE: PuzzleError = PuzzleError::new_const("Puzzle already done");
 
impl <'a, const DIM: usize, U, P, S, C> DfsSolver<'a, DIM, U, P, S, C>
where U: UInt, P: PuzzleState<DIM, U>, S: Strategy<DIM, U, P>, C: Constraint<DIM, U, P> {
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

    fn apply(&mut self, index: PuzzleIndex<DIM>, value: P::Value, alternatives: S::ActionSet, details: bool) -> Result<(), PuzzleError> {
        if self.is_done() {
            return Err(PUZZLE_ALREADY_DONE);
        }
        self.puzzle.apply(index, value)?;
        self.stack.push((index, alternatives));
        self.violation = self.constraint.check(self.puzzle, details);
        self.state = if self.violation.is_none() {
            DfsSolverState::Advancing
        } else {
            DfsSolverState::Backtracking
        };
        return Ok(());
    }

    pub fn manual_step(&mut self, index: PuzzleIndex<DIM>, value: P::Value, details: bool) -> Result<(), PuzzleError> {
        self.apply(index, value, S::empty_action_set(), details)
    }

    pub fn force_backtrack(&mut self) -> bool {
        if self.state == DfsSolverState::Exhausted {
            return false;
        }
        self.state = DfsSolverState::Backtracking;
        true
    }

    pub fn step(&mut self, details: bool) -> Result<(), PuzzleError> {
        match self.state {
            DfsSolverState::Solved => Err(PUZZLE_ALREADY_DONE),
            DfsSolverState::Exhausted => Err(PUZZLE_ALREADY_DONE),
            DfsSolverState::Advancing => {
                // Take a new action
                let (index, mut next_actions) = self.strategy.suggest(self.puzzle)?;
                match next_actions.next() {
                    Some(action) => {
                        self.apply(index, action, next_actions, details)?;
                    }
                    None => {
                        self.state = DfsSolverState::Solved;
                    }
                };
                Ok(())
            }
            DfsSolverState::Backtracking => {
                if self.stack.is_empty() {
                    self.state = DfsSolverState::Exhausted;
                    return Ok(());
                }
                // Backtrack, attempting to advance an existing action set
                let (prev_index, mut alternatives) = self.stack.pop().unwrap();
                self.puzzle.undo(prev_index)?;
                match alternatives.next() {
                    Some(action) => {
                        self.apply(prev_index, action, alternatives, details)?;
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
pub struct FindFirstSolution<'a, const DIM: usize, U, P, S, C>(DfsSolver<'a, DIM, U, P, S, C>, bool)
where U: UInt, P: PuzzleState<DIM, U>, S: Strategy<DIM, U, P>, C: Constraint<DIM, U, P>;

impl <'a, const DIM: usize, U, P, S, C> DfsSolverView<DIM, U, P>
for FindFirstSolution<'a, DIM, U, P, S, C>
where U: UInt, P: PuzzleState<DIM, U>, S: Strategy<DIM, U, P>, C: Constraint<DIM, U, P> {
    fn get_state(&self) -> DfsSolverState { self.0.get_state() }
    fn is_done(&self) -> bool { self.0.is_done() }
    fn is_valid(&self) -> bool { self.0.is_valid() }
    fn get_violations(&self) -> ConstraintResult<DIM> { self.0.get_violations() }
    fn get_puzzle(&self) -> &P { self.0.get_puzzle() }
}

impl <'a, const DIM: usize, U, P, S, C> FindFirstSolution<'a, DIM, U, P, S, C>
where U: UInt, P: PuzzleState<DIM, U>, S: Strategy<DIM, U, P>, C: Constraint<DIM, U, P> {
    pub fn new(
        puzzle: &'a mut P,
        strategy: &'a S,
        constraint: &'a C,
        details: bool,
    ) -> Self {
        FindFirstSolution(DfsSolver::new(puzzle, strategy, constraint), details)
    }

    pub fn step(&mut self) -> Result<&dyn DfsSolverView<DIM, U, P>, PuzzleError> {
        self.0.step(self.1)?;
        Ok(&self.0)
    }

    pub fn solve(&mut self) -> Result<Option<&dyn DfsSolverView<DIM, U, P>>, PuzzleError> {
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
pub struct FindAllSolutions<'a, const DIM: usize, U, P, S, C>(DfsSolver<'a, DIM, U, P, S, C>, bool)
where U: UInt, P: PuzzleState<DIM, U>, S: Strategy<DIM, U, P>, C: Constraint<DIM, U, P>;

impl <'a, const DIM: usize, U, P, S, C> DfsSolverView<DIM, U, P>
for FindAllSolutions<'a, DIM, U, P, S, C>
where U: UInt, P: PuzzleState<DIM, U>, S: Strategy<DIM, U, P>, C: Constraint<DIM, U, P> {
    fn get_state(&self) -> DfsSolverState { self.0.get_state() }
    fn is_done(&self) -> bool { self.0.get_state() == DfsSolverState::Exhausted }
    fn is_valid(&self) -> bool { self.0.is_valid() }
    fn get_violations(&self) -> ConstraintResult<DIM> { self.0.get_violations() }
    fn get_puzzle(&self) -> &P { self.0.get_puzzle() }
}

impl <'a, const DIM: usize, U, P, S, C> FindAllSolutions<'a, DIM, U, P, S, C>
where U: UInt, P: PuzzleState<DIM, U>, S: Strategy<DIM, U, P>, C: Constraint<DIM, U, P> {
    pub fn new(
        puzzle: &'a mut P,
        strategy: &'a S,
        constraint: &'a C,
        details: bool,
    ) -> Self {
        FindAllSolutions(DfsSolver::new(puzzle, strategy, constraint), details)
    }

    pub fn step(&mut self) -> Result<&dyn DfsSolverView<DIM, U, P>, PuzzleError> {
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
    use crate::puzzle::PuzzleValue;
    use crate::strategy::{CompositeStrategy, PartialStrategy};
    use super::*;

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct GwValue(pub u8);
    impl PuzzleValue<u8> for GwValue {
        fn cardinality() -> usize { 9 }
        fn from_uint(u: u8) -> Self { GwValue(u) }
        fn to_uint(self) -> u8 { self.0 }
    }

    #[derive(Clone, Debug)]
    struct GwLine {
        digits: [Option<u8>; 8],
    }

    impl GwLine {
        pub fn new() -> Self {
            GwLine {
                digits: [None; 8],
            }
        }
    }

    impl std::fmt::Display for GwLine {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for i in 0..self.digits.len() {
                if let Some(v) = self.digits[i] {
                    write!(f, "{}", v)?;
                } else {
                    write!(f, ".")?;
                }
            }
            Ok(())
        }
    }

    impl PuzzleState<1, u8> for GwLine {
        type Value = GwValue;

        fn reset(&mut self) {
            self.digits = [None; 8];
        }

        fn get(&self, index: PuzzleIndex<1>) -> Option<Self::Value> {
            if index[0] >= 8 {
                return None;
            }
            self.digits[index[0]].map(|v| GwValue(v))
        }

        fn apply(&mut self, index: PuzzleIndex<1>, value: Self::Value) -> Result<(), PuzzleError> {
            if index[0] >= 8 {
                return Err(PuzzleError::new_const("Index out of bounds"));
            }
            if self.digits[index[0]].is_some() {
                return Err(PuzzleError::new_const("Cell already filled"));
            }
            self.digits[index[0]] = Some(value.to_uint());
            Ok(())
        }

        fn undo(&mut self, index: PuzzleIndex<1>) -> Result<(), PuzzleError> {
            if index[0] >= 8 {
                return Err(PuzzleError::new_const("Index out of bounds"));
            }
            if self.digits[index[0]].is_none() {
                return Err(PuzzleError::new_const("No such action to undo"));
            }
            self.digits[index[0]] = None;
            Ok(())
        }
    }

    struct GwLineConstraint {}
    impl Constraint<1, u8, GwLine> for GwLineConstraint {
        fn check(&self, puzzle: &GwLine, details: bool) -> ConstraintResult<1> {
            let mut violations = Vec::new();
            for i in 0..puzzle.digits.len() {
                if puzzle.digits[i].is_none() {
                    continue;
                }
                let i_val = puzzle.digits[i].unwrap();
                for j in i+1..puzzle.digits.len() {
                    if puzzle.digits[j].is_none() {
                        continue;
                    }
                    let j_val = puzzle.digits[j].unwrap();
                    if i_val == j_val {
                        if details {
                            violations.push(ConstraintViolationDetail {
                                message: format!("Digits with duplicate value: [{}] == [{}] == {}", i, j, i_val),
                                highlight: Some(vec![[i], [j]]),
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
                                highlight: Some(vec![[i], [j]]),
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
    impl Strategy<1, u8, GwLine> for GwLineStrategy {
        type ActionSet = std::vec::IntoIter<GwValue>;

        fn suggest(&self, puzzle: &GwLine) -> Result<(PuzzleIndex<1>, Self::ActionSet), PuzzleError> {
            for i in 0..puzzle.digits.len() {
                if puzzle.digits[i].is_none() {
                    return Ok(([i], vec![1, 2, 3, 4, 5, 6, 7, 8, 9].into_iter().map(GwValue).collect::<Vec<_>>().into_iter()));
                }
            }
            Ok(([0], Self::empty_action_set()))
        }

        fn empty_action_set() -> Self::ActionSet {
            vec![].into_iter()
        }
    }

    struct GwLineStrategyPartial {}
    impl PartialStrategy<1, u8, GwLine> for GwLineStrategyPartial {
        fn suggest_partial(&self, puzzle: &GwLine) -> Result<(PuzzleIndex<1>, Vec<GwValue>), PuzzleError> {
            // This is a partial strategy that only affects the first digit and
            // avoids guessing things that can't work.
            if puzzle.digits[0].is_none() {
                return Ok(([0], vec![4, 6].into_iter().map(GwValue).collect::<Vec<_>>()));
            }
            Ok(([0], vec![]))
        }
    }

    #[test]
    fn german_whispers_constraint() {
        let mut puzzle = GwLine::new();
        let constraint = GwLineConstraint {};
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::NoViolation);
        puzzle.digits[0] = Some(1);
        puzzle.digits[3] = Some(2);
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::NoViolation);
        puzzle.digits[5] = Some(1);
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::Simple("Duplicate digits"));
        puzzle.digits[5] = None;
        puzzle.digits[1] = Some(3);
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::Simple("Adjacent digits too close"));
        puzzle.digits[1] = Some(6);
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::NoViolation);
    }

    #[test]
    fn german_whispers_find() -> Result<(), PuzzleError> {
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
    fn german_whispers_trace() -> Result<(), PuzzleError> {
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
    fn german_whispers_all() -> Result<(), PuzzleError> {
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
    fn german_whispers_all_fast() -> Result<(), PuzzleError> {
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