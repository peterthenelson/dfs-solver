use std::fmt::Debug;

#[derive(Debug)]
pub enum PuzzleError {
    BadAction(String),
    ConstraintViolation(String),
    Other(String),
}

/// Trait for representing solve-state.
pub trait PuzzleState<A>: Clone + Debug {
    fn reset(&mut self);
    fn apply(&mut self, action: &A) -> Result<(), PuzzleError>;
    fn undo(&mut self, action: &A) -> Result<(), PuzzleError>;
}

/// Trait for checking that the current solve-state is valid.
pub trait Constraint<A, P:PuzzleState<A>> {
    fn check(&self, puzzle: &P) -> Result<(), PuzzleError>;
}

/// Train for enumerating the available actions at a particular solve-state
/// (preferably in an order that leads to a faster solve).
pub trait Strategy<A, P:PuzzleState<A>> {
    fn suggest(&self, puzzle: &P) -> Result<Vec<A>, PuzzleError>;
}

/// DFS solver, parameterized by the type of action and state.
pub struct DfsSolver<'a, A, P:PuzzleState<A>, S:Strategy<A, P>> {
    puzzle: &'a mut P,
    strategy: &'a S,
    constraints: Vec<&'a dyn Constraint<A, P>>,
    violations: Vec<PuzzleError>,
    // TODO: Iterators instead of vecs for action sets
    stack: Vec<(usize, Vec<A>)>,
    done: bool,
}

// TODO: Wrapping functions for AnySolution and AllSolutions.

impl <'a, A: Clone + Debug, P:PuzzleState<A>, S:Strategy<A, P>> DfsSolver<'a, A, P, S> {
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
            done: false,
        }
    }

    fn apply(&mut self, index: usize, actions: Vec<A>) -> Result<(), PuzzleError> {
        if self.done {
            return Err(PuzzleError::Other("Solver is done".to_string()));
        } else if index >= actions.len() {
            return Err(PuzzleError::BadAction("Index out of bounds".to_string()));
        }
        self.puzzle.apply(&actions[index])?;
        self.stack.push((index, actions));
        let mut new_violations: Vec<PuzzleError> = Vec::new();
        for constraint in &self.constraints {
            match constraint.check(self.puzzle) {
                Ok(_) => {}
                Err(e) => {
                    new_violations.push(e);
                }
            }
        }
        self.violations = new_violations;
        return Ok(());
    }

    // TODO: Break each unwinding into a separate step
    pub fn step(&mut self) -> Result<(), PuzzleError> {
        if self.done {
            return Ok(());
        }
        if self.violations.is_empty() {
            // Take a new action
            let next_actions = self.strategy.suggest(self.puzzle)?;
            if next_actions.len() == 0 {
                self.done = true;
            } else {
                self.apply(0, next_actions)?;
            }
            return Ok(());
        } else { 
            // Backtrack, attempting to advance an existing or start a new action set
            while self.stack.len() > 0 {
                let (index, actions) = self.stack.pop().unwrap();
                self.puzzle.undo(&actions[index])?;
                if index + 1 < actions.len() {
                    self.apply(index + 1, actions)?;
                    return Ok(());
                }
            }
            self.done = true;
            return Ok(());
        }
    }

    pub fn is_complete(&self) -> bool {
        self.done
    }

    pub fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn get_puzzle(&self) -> &P {
        return &self.puzzle;
    }

    pub fn get_violations(&self) -> &[PuzzleError] {
        return &self.violations;
    }

    pub fn reset(&mut self) {
        self.puzzle.reset();
        self.violations.clear();
        self.stack.clear();
        self.done = false;
    }

    // TODO: Force action fn
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
                return Err(PuzzleError::BadAction("Line is full".to_string()));
            }
        }

        fn undo(&mut self, action: &u8) -> Result<(), PuzzleError> {
            if self.digits.len() > 0 {
                if self.digits.last() != Some(action) {
                    return Err(PuzzleError::BadAction("Action not found".to_string()));
                }
                self.digits.pop();
                return Ok(())
            } else {
                return Err(PuzzleError::BadAction("No actions to undo".to_string()));
            }
        }
    }

    struct GwLineConstraint {}
    impl Constraint<u8, GwLine> for GwLineConstraint {
        fn check(&self, puzzle: &GwLine) -> Result<(), PuzzleError> {
            for i in 0..puzzle.digits.len() {
                for j in i+1..puzzle.digits.len() {
                    if puzzle.digits[i] == puzzle.digits[j] {
                        return Err(PuzzleError::ConstraintViolation("Duplicate digits found".to_string()));
                    }
                    let diff: i16 = (puzzle.digits[i] as i16) - (puzzle.digits[j] as i16);
                    if j == i+1 && diff.abs() < 5 {
                        return Err(PuzzleError::ConstraintViolation("Digits too close".to_string()));
                    }
                }
            }
            Ok(())
        }
    }

    struct GwLineStrategy {}
    impl Strategy<u8, GwLine> for GwLineStrategy {
        fn suggest(&self, puzzle: &GwLine) -> Result<Vec<u8>, PuzzleError> {
            if puzzle.full() {
                return Ok(vec![]);
            }
            return Ok(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
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
        while !solver.is_complete() {
            match solver.step() {
                Ok(_) => {
                    print!("Current state: {:?}\n", solver.get_puzzle());
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