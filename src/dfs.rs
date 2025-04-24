use std::fmt::Debug;
use std::borrow::Cow;

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

/// Violation of a constraint. Optionally provides a list of actions that are
/// relevant to the violation.
/// TODO: Rethink how to do the detailed-vs-not reporting to avoid allocations
/// when not required.
#[derive(Debug, Clone)]
pub struct ConstraintViolation<A> {
    pub message: String,
    pub highlight: Option<Vec<A>>,
}

/// Trait for checking that the current solve-state is valid.
pub trait Constraint<A, P> where P: PuzzleState<A> {
    fn check(&self, puzzle: &P, details: bool) -> Option<ConstraintViolation<A>>;
}

/// Trait for enumerating the available actions at a particular decision point
/// (preferably in an order that leads to a faster solve). Note that the
/// posibilities must be exhaustive (e.g., if the puzzle is a sudoku, the first
/// empty cell must be one of the 9 digits--if none of them work, then the
/// puzzle has no solution).
pub trait Strategy<A, P> where P: PuzzleState<A> {
    type ActionSet: Iterator<Item = A>;
    fn suggest(&self, puzzle: &P) -> Result<Self::ActionSet, PuzzleError>;
    fn empty_action_set() -> Self::ActionSet;
}

/// The state of the DFS solver. At any point in time, the solver is either
/// advancing (ready to take a new action), backtracking (undoing actions),
/// solved (has found a solution), or exhausted (no more actions to take).
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DfsSolverState {
    Advancing,
    Backtracking,
    Solved,
    Exhausted,
}

// A view on the state and associated data for the solver.
pub trait DfsSolverView<A, P> where A:Clone + Debug, P: PuzzleState<A> {
    fn get_state(&self) -> DfsSolverState;
    fn is_done(&self) -> bool;
    fn is_valid(&self) -> bool;
    fn get_violations(&self) -> &[ConstraintViolation<A>];
    fn get_puzzle(&self) -> &P;
}

/// DFS solver. If you want a lower-level API that allows for more control over
/// the solving process, you can directly use this. Most users should prefer
/// FindFirstSolution or FindAllSolutions, which are higher-level APIs. However,
/// if you are implementing a UI or debugging, this API may be useful.
pub struct DfsSolver<'a, A, P, S> where A:Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P> {
    puzzle: &'a mut P,
    strategy: &'a S,
    constraints: Vec<&'a dyn Constraint<A, P>>,
    violations: Vec<ConstraintViolation<A>>,
    stack: Vec<(A, <S as Strategy<A, P>>::ActionSet)>,
    state: DfsSolverState,
}

impl <'a, A, P, S> DfsSolverView<A, P> for DfsSolver<'a, A, P, S> where A:Clone + Debug, P:PuzzleState<A>, S:Strategy<A, P> {
    fn get_state(&self) -> DfsSolverState {
        self.state
    }

    fn is_done(&self) -> bool {
        self.state == DfsSolverState::Solved || self.state == DfsSolverState::Exhausted
    }

    fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }

    fn get_violations(&self) -> &[ConstraintViolation<A>] {
        &self.violations
    }

    fn get_puzzle(&self) -> &P {
        self.puzzle
    }
}

const PUZZLE_ALREADY_DONE: PuzzleError = PuzzleError::new_const("Puzzle already done");

impl <'a, A, P, S> DfsSolver<'a, A, P, S> where A:Clone + Debug, P:PuzzleState<A>, S:Strategy<A, P> {
    pub fn new(
        puzzle: &'a mut P,
        strategy: &'a S,
        constraints: Vec<&'a dyn Constraint<A, P>>,
    ) -> Self {
        DfsSolver {
            puzzle,
            strategy,
            constraints,
            violations: Vec::new(),
            stack: Vec::new(),
            state: DfsSolverState::Advancing,
        }
    }

    fn apply(&mut self, action: A, alternatives: S::ActionSet, detail: bool) -> Result<(), PuzzleError> {
        if self.is_done() {
            return Err(PUZZLE_ALREADY_DONE);
        }
        self.puzzle.apply(&action)?;
        self.stack.push((action, alternatives));
        self.violations = self.constraints.iter().filter_map(|c| c.check(self.puzzle, detail)).collect();
        self.state = if self.violations.is_empty() {
            DfsSolverState::Advancing
        } else {
            DfsSolverState::Backtracking
        };
        return Ok(());
    }

    pub fn manual_step(&mut self, action: A, detail: bool) -> Result<(), PuzzleError> {
        self.apply(action, S::empty_action_set(), detail)
    }

    pub fn force_backtrack(&mut self) -> bool {
        if self.state == DfsSolverState::Exhausted {
            return false;
        }
        self.state = DfsSolverState::Backtracking;
        true
    }

    pub fn step(&mut self, detail: bool) -> Result<(), PuzzleError> {
        match self.state {
            DfsSolverState::Solved => Err(PUZZLE_ALREADY_DONE),
            DfsSolverState::Exhausted => Err(PUZZLE_ALREADY_DONE),
            DfsSolverState::Advancing => {
                // Take a new action
                let mut next_actions = self.strategy.suggest(self.puzzle)?;
                match next_actions.next() {
                    Some(action) => {
                        self.apply(action, next_actions, detail)?;
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
                let (prev_action, mut alternatives) = self.stack.pop().unwrap();
                self.puzzle.undo(&prev_action)?;
                match alternatives.next() {
                    Some(action) => {
                        self.apply(action, alternatives, detail)?;
                        Ok(())
                    }
                    None => Ok(()),
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.puzzle.reset();
        self.violations.clear();
        self.stack.clear();
        self.state = DfsSolverState::Advancing;
    }
}

/// Find first solution to the puzzle using the given strategy and constraints.
pub struct FindFirstSolution<'a, A, P, S>(DfsSolver<'a, A, P, S>, bool) where A:Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P>;

impl <'a, A, P, S> DfsSolverView<A, P> for FindFirstSolution<'a, A, P, S> where A:Clone + Debug, P:PuzzleState<A>, S:Strategy<A, P> {
    fn get_state(&self) -> DfsSolverState { self.0.get_state() }
    fn is_done(&self) -> bool { self.0.is_done() }
    fn is_valid(&self) -> bool { self.0.is_valid() }
    fn get_violations(&self) -> &[ConstraintViolation<A>] { self.0.get_violations() }
    fn get_puzzle(&self) -> &P { self.0.get_puzzle() }
}

impl <'a, A, P, S> FindFirstSolution<'a, A, P, S> where A:Clone + Debug, P:PuzzleState<A>, S:Strategy<A, P> {
    pub fn new(
        puzzle: &'a mut P,
        strategy: &'a S,
        constraints: Vec<&'a dyn Constraint<A, P>>,
        detail: bool,
    ) -> Self {
        FindFirstSolution(DfsSolver::new(puzzle, strategy, constraints), detail)
    }

    pub fn step(&mut self) -> Result<&dyn DfsSolverView<A, P>, PuzzleError> {
        self.0.step(self.1)?;
        Ok(&self.0)
    }

    pub fn solve(&mut self) -> Result<Option<&dyn DfsSolverView<A, P>>, PuzzleError> {
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
pub struct FindAllSolutions<'a, A, P, S>(DfsSolver<'a, A, P, S>, bool) where A:Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P>;

impl <'a, A, P, S> DfsSolverView<A, P> for FindAllSolutions<'a, A, P, S> where A:Clone + Debug, P:PuzzleState<A>, S:Strategy<A, P> {
    fn get_state(&self) -> DfsSolverState { self.0.get_state() }
    fn is_done(&self) -> bool { self.0.get_state() == DfsSolverState::Exhausted }
    fn is_valid(&self) -> bool { self.0.is_valid() }
    fn get_violations(&self) -> &[ConstraintViolation<A>] { self.0.get_violations() }
    fn get_puzzle(&self) -> &P { self.0.get_puzzle() }
}

impl <'a, A, P, S> FindAllSolutions<'a, A, P, S> where A:Clone + Debug, P:PuzzleState<A>, S:Strategy<A, P> {
    pub fn new(
        puzzle: &'a mut P,
        strategy: &'a S,
        constraints: Vec<&'a dyn Constraint<A, P>>,
        detail: bool,
    ) -> Self {
        FindAllSolutions(DfsSolver::new(puzzle, strategy, constraints), detail)
    }

    pub fn step(&mut self) -> Result<&dyn DfsSolverView<A, P>, PuzzleError> {
        if self.0.state == DfsSolverState::Solved {
            self.0.force_backtrack();
        }
        self.0.step(self.1)?;
        Ok(&self.0)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Clone, Debug)]
    struct GwLine {
        digits: Vec<u8>,
    }

    impl GwLine {
        pub fn new() -> Self {
            GwLine {
                digits: Vec::new(),
            }
        }

        fn full(&self) -> bool {
            self.digits.len() == 8
        }
    }

    impl PuzzleState<u8> for GwLine {
        fn reset(&mut self) {
            self.digits.clear();
        }

        fn apply(&mut self, action: &u8) -> Result<(), PuzzleError> {
            if !self.full() {
                self.digits.push(*action);
                return Ok(())
            } else {
                return Err(PuzzleError::new_const("Line is full"));
            }
        }

        fn undo(&mut self, action: &u8) -> Result<(), PuzzleError> {
            if self.digits.len() > 0 {
                if self.digits.last() != Some(action) {
                    return Err(PuzzleError::new_const("Action not found"));
                }
                self.digits.pop();
                return Ok(())
            } else {
                return Err(PuzzleError::new_const("No actions to undo"));
            }
        }
    }

    struct GwLineConstraint {}
    impl Constraint<u8, GwLine> for GwLineConstraint {
        fn check(&self, puzzle: &GwLine, detail: bool) -> Option<ConstraintViolation<u8>> {
            for i in 0..puzzle.digits.len() {
                for j in i+1..puzzle.digits.len() {
                    let highlight = if detail {
                        Some(vec![puzzle.digits[i], puzzle.digits[j]])
                    } else {
                        None
                    };
                    if puzzle.digits[i] == puzzle.digits[j] {
                        return Some(ConstraintViolation {
                            message: "Duplicate digits".to_string(),
                            highlight,
                        });
                    }
                    let diff: i16 = (puzzle.digits[i] as i16) - (puzzle.digits[j] as i16);
                    if j == i+1 && diff.abs() < 5 {
                        return Some(ConstraintViolation {
                            message: "Digits too close".to_string(),
                            highlight,
                        });
                    }
                }
            }
            None
        }
    }

    struct GwLineStrategy {}
    impl Strategy<u8, GwLine> for GwLineStrategy {
        type ActionSet = std::vec::IntoIter<u8>;
        fn suggest(&self, puzzle: &GwLine) -> Result<Self::ActionSet, PuzzleError> {
            if puzzle.full() {
                return Ok(vec![].into_iter());
            }
            return Ok(vec![1, 2, 3, 4, 5, 6, 7, 8, 9].into_iter());
        }
        fn empty_action_set() -> Self::ActionSet {
            vec![].into_iter()
        }
    }

    #[test]
    fn german_whispers_find() -> Result<(), PuzzleError> {
        let mut puzzle = GwLine::new();
        let strategy = GwLineStrategy {};
        let constraint = GwLineConstraint {};
        let mut finder = FindFirstSolution::new(&mut puzzle, &strategy, vec![&constraint], false);
        let maybe_solution = finder.solve()?;
        assert!(maybe_solution.is_some());
        println!("  Solution: {:?}", maybe_solution.unwrap().get_puzzle());
        Ok(())
    }

    #[test]
    fn german_whispers_trace() -> Result<(), PuzzleError> {
        let mut puzzle = GwLine::new();
        let strategy = GwLineStrategy {};
        let constraint = GwLineConstraint {};
        let mut finder = FindFirstSolution::new(&mut puzzle, &strategy, vec![&constraint], true);
        let mut steps: usize = 0;
        let mut violation_count: usize = 0;
        while !finder.is_done() {
            finder.step()?;
            steps += 1;
            violation_count += finder.get_violations().len();
        }
        assert!(finder.is_valid());
        println!("  Steps: {}", steps);
        println!("  Violations: {}", violation_count);
        Ok(())
    }

    #[test]
    fn german_whispers_all() -> Result<(), PuzzleError> {
        let mut puzzle = GwLine::new();
        let strategy = GwLineStrategy {};
        let constraint = GwLineConstraint {};
        let mut finder = FindAllSolutions::new(&mut puzzle, &strategy, vec![&constraint], false);
        let mut solution_count: usize = 0;
        while !finder.is_done() {
            finder.step()?;
            solution_count += if finder.get_state() == DfsSolverState::Solved { 1 } else { 0 };
        }
        println!("  Num solutions: {}", solution_count);
        Ok(())
    }

}