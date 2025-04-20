use std::fmt::Debug;

/// Error types for the solver. These indicate something wrong with either the
/// puzzle/strategy/constraints or with the algorithm itself.
#[derive(Debug, Clone)]
pub struct PuzzleError(String);

/// Trait for representing solve-state.
pub trait PuzzleState<A>: Clone + Debug {
    fn reset(&mut self);
    fn apply(&mut self, action: &A) -> Result<(), PuzzleError>;
    fn undo(&mut self, action: &A) -> Result<(), PuzzleError>;
}

/// Violation of a constraint. Optionally provides a list of actions that are
/// relevant to the violation.
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
/// empty cell must be one of the 9 digits--there aren't any other possibilities).
pub trait Strategy<A, P> where P: PuzzleState<A> {
    type ActionSet: Iterator<Item = A>;
    fn suggest(&self, puzzle: &P) -> Result<Self::ActionSet, PuzzleError>;
    fn empty_action_set() -> Self::ActionSet;
}

/// The state of the DFS solver. This is used to track the current state of the
/// solver and whether it is advancing (ready to take new actions), backtracking
/// (undoing actions), or done (no more actions to take).
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DfsSolverState {
    Advancing,
    Backtracking,
    Done,
}

/// DFS solver. This is a lower-level API that allows for more control over
/// the solving process. It is not recommended for most users, but is useful
/// when additional control is needed for debugging or UI purposes. See the
/// wrapper functions for a higher-level API.
pub struct DfsSolver<'a, A, P, S> where A:Clone + Debug, P: PuzzleState<A>, S: Strategy<A, P> {
    puzzle: &'a mut P,
    strategy: &'a S,
    constraints: Vec<&'a dyn Constraint<A, P>>,
    violations: Vec<ConstraintViolation<A>>,
    stack: Vec<(A, <S as Strategy<A, P>>::ActionSet)>,
    state: DfsSolverState,
}

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
        if self.state == DfsSolverState::Done {
            return Err(PuzzleError("Solver is done".to_string()));
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

    pub fn step(&mut self, detail: bool) -> Result<(), PuzzleError> {
        match self.state {
            DfsSolverState::Done => Ok(()),
            DfsSolverState::Advancing => {
                // Take a new action
                let mut next_actions = self.strategy.suggest(self.puzzle)?;
                match next_actions.next() {
                    Some(action) => {
                        self.apply(action, next_actions, detail)?;
                    }
                    None => {
                        self.state = DfsSolverState::Done;
                    }
                };
                Ok(())
            }
            DfsSolverState::Backtracking => {
                if self.stack.is_empty() {
                    self.state = DfsSolverState::Done;
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

    pub fn get_state(&self) -> DfsSolverState {
        self.state
    }

    pub fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn get_puzzle(&self) -> &P {
        return &self.puzzle;
    }

    pub fn get_violations(&self) -> &[ConstraintViolation<A>] {
        return &self.violations;
    }

    pub fn reset(&mut self) {
        self.puzzle.reset();
        self.violations.clear();
        self.stack.clear();
        self.state = DfsSolverState::Advancing;
    }
}

// TODO: Wrapping functions for AnySolution and AllSolutions.

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
                return Err(PuzzleError("Line is full".to_string()));
            }
        }

        fn undo(&mut self, action: &u8) -> Result<(), PuzzleError> {
            if self.digits.len() > 0 {
                if self.digits.last() != Some(action) {
                    return Err(PuzzleError("Action not found".to_string()));
                }
                self.digits.pop();
                return Ok(())
            } else {
                return Err(PuzzleError("No actions to undo".to_string()));
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
    fn german_whispers() {
        let mut puzzle = GwLine::new();
        let strategy = GwLineStrategy {};
        let constraint = GwLineConstraint {};
        let mut solver = DfsSolver::new(
            &mut puzzle, &strategy, vec![&constraint],
        );
        while solver.get_state() != DfsSolverState::Done {
            match solver.step(true) {
                Ok(_) => {
                    print!("Current state: {:?} -- {:?}\n", solver.get_state(), solver.get_puzzle());
                }
                Err(e) => {
                    println!("Error: {:?}", e);
                    break;
                }
            }
        }
        if solver.is_valid() {
            println!("Solution found: {:?}", puzzle);
        } else {
            println!("No solution found.");
        }
    }
}