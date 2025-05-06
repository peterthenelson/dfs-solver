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

/// Potential violation of a constraint. The optimized approach to use in the
/// solver is to return the first violation found without any detailed
/// information. However, for debugging or UI purposes, it may be useful to
/// return a list of all the violations, along with the specific actions that
/// caused them.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintResult<A> {
    Simple(&'static str),
    Details(Vec<ConstraintViolationDetail<A>>),
    NoViolation,
}

impl <A> ConstraintResult<A> {
    pub fn is_none(&self) -> bool {
        match self {
            ConstraintResult::NoViolation => true,
            _ => false,
        }
    }
}

pub struct ConstraintConjunction<A, P, X, Y>
where A: Clone + Debug, P: PuzzleState<A>, X: Constraint<A, P>, Y: Constraint<A, P> {
    pub x: X,
    pub y: Y,
    pub p_a: std::marker::PhantomData<A>,
    pub p_p: std::marker::PhantomData<P>,
}

impl <A, P, X, Y> ConstraintConjunction<A, P, X, Y>
where A: Clone + Debug, P: PuzzleState<A>, X: Constraint<A, P>, Y: Constraint<A, P> {
    pub fn new(x: X, y: Y) -> Self {
        ConstraintConjunction { x, y, p_a: std::marker::PhantomData, p_p: std::marker::PhantomData }
    }
}

impl <A, P, X, Y> Constraint<A, P> for ConstraintConjunction<A, P, X, Y>
where A: Clone + Debug, P: PuzzleState<A>, X: Constraint<A, P>, Y: Constraint<A, P> {
    fn check(&self, puzzle: &P, details: bool) -> ConstraintResult<A> {
        if details {
            let mut violations = Vec::new();
            match self.x.check(puzzle, details) {
                ConstraintResult::NoViolation => (),
                ConstraintResult::Simple(message) => {
                    violations.push(ConstraintViolationDetail {
                        message: message.to_string(),
                        highlight: None,
                    });
                },
                ConstraintResult::Details(mut violation) => {
                    violations.append(&mut violation);
                }
            };
            match self.y.check(puzzle, details) {
                ConstraintResult::NoViolation => (),
                ConstraintResult::Simple(message) => {
                    violations.push(ConstraintViolationDetail {
                        message: message.to_string(),
                        highlight: None,
                    });
                },
                ConstraintResult::Details(mut violation) => {
                    violations.append(&mut violation);
                }
            };
            ConstraintResult::Details(violations)
        } else {
            match self.x.check(puzzle, details) {
                ConstraintResult::NoViolation => self.y.check(puzzle, details),
                x_result => x_result,
            }
        }
    }
}


#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintViolationDetail<A> {
    pub message: String,
    pub highlight: Option<Vec<A>>,
}

/// Trait for checking that the current solve-state is valid.
pub trait Constraint<A, P> where P: PuzzleState<A> {
    fn check(&self, puzzle: &P, details: bool) -> ConstraintResult<A>;
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

/// A partial strategy that can be used to suggest actions (but requires some
/// other strategy to fall back on). This is useful for strategies that are
/// specific to a particular constraint or puzzle type, but may not be able to
/// enumerate all the possible actions.
pub trait PartialStrategy<A, P>: where P: PuzzleState<A> {
    fn suggest(&self, puzzle: &P) -> Result<Vec<A>, PuzzleError>;
}

/// All strategies are partial strategies.
impl <A, P, S> PartialStrategy<A, P> for S
where A: Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P> {
    fn suggest(&self, puzzle: &P) -> Result<Vec<A>, PuzzleError> {
        let mut actions = Vec::new();
        for action in self.suggest(puzzle)? {
            actions.push(action);
        }
        Ok(actions)
    }
}

/// A composite strategy that combines multiple strategies into one.
pub struct CompositeStrategy<'a, A, P, S>
where A: Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P> {
    default_strategy: S,
    partial_strategies: Vec<&'a dyn PartialStrategy<A, P>>,
    p_a: std::marker::PhantomData<A>,
    p_p: std::marker::PhantomData<P>,
}

impl <'a, A, P, S> CompositeStrategy<'a, A, P, S>
where A: Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P> {
    pub fn new(default_strategy: S, partial_strategies: Vec<&'a dyn PartialStrategy<A, P>>) -> Self {
        CompositeStrategy {
            default_strategy,
            partial_strategies,
            p_a: std::marker::PhantomData,
            p_p: std::marker::PhantomData,
        }
    }
}

impl <'a, A, P, S> Strategy<A, P> for CompositeStrategy<'a, A, P, S>
where A: Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P> {
    type ActionSet = std::vec::IntoIter<A>;

    fn suggest(&self, puzzle: &P) -> Result<Self::ActionSet, PuzzleError> {
        let mut actions = Vec::new();
        for strategy in &self.partial_strategies {
            actions.extend(strategy.suggest(puzzle)?);
            if !actions.is_empty() {
                return Ok(actions.into_iter());
            }
        }
        actions.extend(self.default_strategy.suggest(puzzle)?);
        return Ok(actions.into_iter());
    }

    fn empty_action_set() -> Self::ActionSet {
        vec![].into_iter()
    }
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
pub trait DfsSolverView<A, P> where A: Clone + Debug, P: PuzzleState<A> {
    fn get_state(&self) -> DfsSolverState;
    fn is_done(&self) -> bool;
    fn is_valid(&self) -> bool;
    fn get_violations(&self) -> ConstraintResult<A>;
    fn get_puzzle(&self) -> &P;
}

/// DFS solver. If you want a lower-level API that allows for more control over
/// the solving process, you can directly use this. Most users should prefer
/// FindFirstSolution or FindAllSolutions, which are higher-level APIs. However,
/// if you are implementing a UI or debugging, this API may be useful.
pub struct DfsSolver<'a, A, P, S, C>
where A: Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P>, C: Constraint<A, P> {
    puzzle: &'a mut P,
    strategy: &'a S,
    constraint: &'a C,
    violation: ConstraintResult<A>,
    stack: Vec<(A, <S as Strategy<A, P>>::ActionSet)>,
    state: DfsSolverState,
}

impl <'a, A, P, S, C> DfsSolverView<A, P> for DfsSolver<'a, A, P, S, C>
where A: Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P>, C: Constraint<A, P> {
    fn get_state(&self) -> DfsSolverState {
        self.state
    }

    fn is_done(&self) -> bool {
        self.state == DfsSolverState::Solved || self.state == DfsSolverState::Exhausted
    }

    fn is_valid(&self) -> bool {
        self.violation.is_none()
    }

    fn get_violations(&self) -> ConstraintResult<A> {
        self.violation.clone()
    }

    fn get_puzzle(&self) -> &P {
        self.puzzle
    }
}

const PUZZLE_ALREADY_DONE: PuzzleError = PuzzleError::new_const("Puzzle already done");

impl <'a, A, P, S, C> DfsSolver<'a, A, P, S, C>
where A: Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P>, C: Constraint<A, P> {
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

    fn apply(&mut self, action: A, alternatives: S::ActionSet, details: bool) -> Result<(), PuzzleError> {
        if self.is_done() {
            return Err(PUZZLE_ALREADY_DONE);
        }
        self.puzzle.apply(&action)?;
        self.stack.push((action, alternatives));
        self.violation = self.constraint.check(self.puzzle, details);
        self.state = if self.violation.is_none() {
            DfsSolverState::Advancing
        } else {
            DfsSolverState::Backtracking
        };
        return Ok(());
    }

    pub fn manual_step(&mut self, action: A, details: bool) -> Result<(), PuzzleError> {
        self.apply(action, S::empty_action_set(), details)
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
                let mut next_actions = self.strategy.suggest(self.puzzle)?;
                match next_actions.next() {
                    Some(action) => {
                        self.apply(action, next_actions, details)?;
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
                        self.apply(action, alternatives, details)?;
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
pub struct FindFirstSolution<'a, A, P, S, C>(DfsSolver<'a, A, P, S, C>, bool)
where A: Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P>, C: Constraint<A, P>;

impl <'a, A, P, S, C> DfsSolverView<A, P> for FindFirstSolution<'a, A, P, S, C>
where A: Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P>, C: Constraint<A, P> {
    fn get_state(&self) -> DfsSolverState { self.0.get_state() }
    fn is_done(&self) -> bool { self.0.is_done() }
    fn is_valid(&self) -> bool { self.0.is_valid() }
    fn get_violations(&self) -> ConstraintResult<A> { self.0.get_violations() }
    fn get_puzzle(&self) -> &P { self.0.get_puzzle() }
}

impl <'a, A, P, S, C> FindFirstSolution<'a, A, P, S, C>
where A: Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P>, C: Constraint<A, P> {
    pub fn new(
        puzzle: &'a mut P,
        strategy: &'a S,
        constraint: &'a C,
        details: bool,
    ) -> Self {
        FindFirstSolution(DfsSolver::new(puzzle, strategy, constraint), details)
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
pub struct FindAllSolutions<'a, A, P, S, C>(DfsSolver<'a, A, P, S, C>, bool)
    where A: Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P>, C: Constraint<A, P>;

impl <'a, A, P, S, C> DfsSolverView<A, P> for FindAllSolutions<'a, A, P, S, C>
where A: Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P>, C: Constraint<A, P> {
    fn get_state(&self) -> DfsSolverState { self.0.get_state() }
    fn is_done(&self) -> bool { self.0.get_state() == DfsSolverState::Exhausted }
    fn is_valid(&self) -> bool { self.0.is_valid() }
    fn get_violations(&self) -> ConstraintResult<A> { self.0.get_violations() }
    fn get_puzzle(&self) -> &P { self.0.get_puzzle() }
}

impl <'a, A, P, S, C> FindAllSolutions<'a, A, P, S, C>
where A: Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P>, C: Constraint<A, P> {
    pub fn new(
        puzzle: &'a mut P,
        strategy: &'a S,
        constraint: &'a C,
        details: bool,
    ) -> Self {
        FindAllSolutions(DfsSolver::new(puzzle, strategy, constraint), details)
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

        fn len(&self) -> usize {
            self.digits.len()
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
        fn check(&self, puzzle: &GwLine, details: bool) -> ConstraintResult<u8> {
            let mut violations = Vec::new();
            for i in 0..puzzle.digits.len() {
                for j in i+1..puzzle.digits.len() {
                    if puzzle.digits[i] == puzzle.digits[j] {
                        if details {
                            violations.push(ConstraintViolationDetail {
                                message: format!("Duplicate digits: {} and {}", puzzle.digits[i], puzzle.digits[j]),
                                highlight: Some(vec![puzzle.digits[i], puzzle.digits[j]]),
                            });
                        } else {
                            return ConstraintResult::Simple("Duplicate digits");
                        }
                    }
                    let diff: i16 = (puzzle.digits[i] as i16) - (puzzle.digits[j] as i16);
                    if j == i+1 && diff.abs() < 5 {
                        if details {
                            violations.push(ConstraintViolationDetail {
                                message: format!("Adjacent digits too close: {} and {}", puzzle.digits[i], puzzle.digits[j]),
                                highlight: Some(vec![puzzle.digits[i], puzzle.digits[j]]),
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

    struct GwLineStrategyPartial {}
    impl PartialStrategy<u8, GwLine> for GwLineStrategyPartial {
        fn suggest(&self, puzzle: &GwLine) -> Result<Vec<u8>, PuzzleError> {
            // This is a partial strategy that only affects the first digit and
            // avoids guessing things that can't work.
            if puzzle.len() == 0 {
                return Ok(vec![4, 6]);
            }
            Ok(vec![])
        }
    }

    #[test]
    fn german_whispers_find() -> Result<(), PuzzleError> {
        let mut puzzle = GwLine::new();
        let strategy = GwLineStrategy {};
        let constraint = GwLineConstraint {};
        let mut finder = FindFirstSolution::new(&mut puzzle, &strategy, &constraint, false);
        let maybe_solution = finder.solve()?;
        assert!(maybe_solution.is_some());
        assert_eq!(maybe_solution.unwrap().get_puzzle().digits, vec![4, 9, 3, 8, 2, 7, 1, 6]);
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