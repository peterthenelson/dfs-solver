use std::fmt::Debug;
use crate::core::{Error, Index, State, UInt};
use crate::constraint::{Constraint, ConstraintResult, ConstraintViolationDetail};
use crate::strategy::{BranchPoint, Strategy};

/// The state of the DFS solver. At any point in time, the solver is either
/// advancing (ready to take a new action), backtracking (undoing actions),
/// solved (has found a solution), or exhausted (no more actions to take).
#[derive(Debug, PartialEq, Clone, Copy, Eq)]
pub enum DfsSolverState {
    Init,
    Advancing,
    Backtracking,
    Solved,
    Exhausted,
}

// A view on the state and associated data for the solver.
pub trait DfsSolverView<U: UInt, S: State<U>> {
    fn solver_state(&self) -> DfsSolverState;
    fn is_done(&self) -> bool;
    fn is_valid(&self) -> bool;
    fn check_constraints(&self) -> ConstraintResult<U, S::Value>;
    fn explain_contradiction(&self) -> Vec<ConstraintViolationDetail>;
    fn get_state(&self) -> &S;
}

/// DFS solver. If you want a lower-level API that allows for more control over
/// the solving process, you can directly use this. Most users should prefer
/// FindFirstSolution or FindAllSolutions, which are higher-level APIs. However,
/// if you are implementing a UI or debugging, this API may be useful.
pub struct DfsSolver<'a, U, S, St, C>
where U: UInt, S: State<U>, St: Strategy<U, S>, C: Constraint<U, S> {
    puzzle: &'a mut S,
    strategy: &'a St,
    constraint: &'a mut C,
    check_result: ConstraintResult<U, S::Value>,
    stack: Vec<BranchPoint<U, S, St::ActionSet>>,
    state: DfsSolverState,
}

impl <'a, U, S, St, C> Debug
for DfsSolver<'a, U, S, St, C>
where U: UInt, S: State<U>, St: Strategy<U, S>, C: Constraint<U, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State:\n{:?}Constraint:\n{:?}\n", self.puzzle, self.constraint)
    }
}

impl <'a, U, S, St, C> DfsSolverView<U, S>
for DfsSolver<'a, U, S, St, C>
where U: UInt, S: State<U>, St: Strategy<U, S>, C: Constraint<U, S> {
    fn solver_state(&self) -> DfsSolverState {
        self.state
    }

    fn is_done(&self) -> bool {
        self.state == DfsSolverState::Solved || self.state == DfsSolverState::Exhausted
    }

    fn is_valid(&self) -> bool {
        self.check_result.no_contradiction()
    }

    fn check_constraints(&self) -> ConstraintResult<U, S::Value> {
        self.check_result.clone()
    }

    fn explain_contradiction(&self) -> Vec<ConstraintViolationDetail> {
        self.constraint.explain_contradictions(self.get_state())
    }

    fn get_state(&self) -> &S {
        self.puzzle
    }
}

const NOT_INITIALIZED: Error = Error::new_const("Must call init() before stepping forward");
const PUZZLE_ALREADY_DONE: Error = Error::new_const("Puzzle already done");
const NO_CHOICE: Error = Error::new_const("Decision point has no choice");
 
impl <'a, U, S, St, C> DfsSolver<'a, U, S, St, C>
where U: UInt, S: State<U>, St: Strategy<U, S>, C: Constraint<U, S> {
    pub fn new(
        puzzle: &'a mut S,
        strategy: &'a St,
        constraint: &'a mut C, 
    ) -> Self {
        DfsSolver {
            puzzle,
            strategy,
            constraint,
            check_result: ConstraintResult::any(),
            stack: Vec::new(),
            state: DfsSolverState::Init,
        }
    }

    pub fn init(&mut self) -> Result<(), Error> {
        for r in 0..S::ROWS {
            for c in 0..S::COLS {
                if let Some(v) = self.puzzle.get([r, c]) {
                    self.constraint.apply([r, c], v)?;
                }
            }
        }
        self.state = DfsSolverState::Advancing;
        Ok(())
    }

    fn apply(&mut self, decision: BranchPoint<U, S, St::ActionSet>, force_grid: bool) -> Result<(), Error> {
        if self.state == DfsSolverState::Init {
            return Err(NOT_INITIALIZED);
        } else if self.is_done() {
            return Err(PUZZLE_ALREADY_DONE);
        } else if decision.chosen.is_none() {
            return Err(NO_CHOICE);
        }
        {
            let v = decision.chosen.unwrap();
            self.puzzle.apply(decision.index, v)?;
            if let Err(e) = self.constraint.apply(decision.index, v) {
                self.puzzle.undo(decision.index, v)?;
                return Err(e);
            }
        }
        self.stack.push(decision);
        self.check_result = self.constraint.check(self.puzzle, force_grid);
        self.state = if self.check_result.no_contradiction() {
            DfsSolverState::Advancing
        } else {
            DfsSolverState::Backtracking
        };
        return Ok(());
    }

    fn undo(&mut self, decision: &BranchPoint<U, S, St::ActionSet>) -> Result<(), Error> {
        let v = decision.chosen.unwrap();
        if let Err(e) = self.puzzle.undo(decision.index, v) {
            self.constraint.undo(decision.index, v)?;
            return Err(e);
        }
        self.constraint.undo(decision.index, v)
    }

    pub fn manual_step(&mut self, index: Index, value: S::Value, force_grid: bool) -> Result<(), Error> {
        self.apply(BranchPoint {
            chosen: Some(value),
            index,
            alternatives: St::ActionSet::default(),
         }, force_grid)
    }

    pub fn force_backtrack(&mut self) -> bool {
        if self.state == DfsSolverState::Exhausted {
            return false;
        }
        self.state = DfsSolverState::Backtracking;
        true
    }

    pub fn step(&mut self, force_grid: bool) -> Result<(), Error> {
        match self.state {
            DfsSolverState::Init => Err(NOT_INITIALIZED),
            DfsSolverState::Solved => Err(PUZZLE_ALREADY_DONE),
            DfsSolverState::Exhausted => Err(PUZZLE_ALREADY_DONE),
            DfsSolverState::Advancing => {
                // Take a new action
                let decision = self.strategy.suggest(self.puzzle)?;
                if decision.chosen.is_some() {
                    self.apply(decision, force_grid)?;
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
                self.undo(&decision)?;
                match decision.advance() {
                    Some(_) => {
                        self.apply(decision, force_grid)?;
                        Ok(())
                    }
                    None => Ok(()),
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.puzzle.reset();
        self.check_result = ConstraintResult::any();
        self.stack.clear();
        self.state = DfsSolverState::Advancing;
    }
}

/// Find first solution to the puzzle using the given strategy and constraints.
pub struct FindFirstSolution<'a, U, S, St, C>(DfsSolver<'a, U, S, St, C>, bool)
where U: UInt, S: State<U>, St: Strategy<U, S>, C: Constraint<U, S>;

impl <'a, U, S, St, C> DfsSolverView<U, S>
for FindFirstSolution<'a, U, S, St, C>
where U: UInt, S: State<U>, St: Strategy<U, S>, C: Constraint<U, S> {
    fn solver_state(&self) -> DfsSolverState { self.0.solver_state() }
    fn is_done(&self) -> bool { self.0.is_done() }
    fn is_valid(&self) -> bool { self.0.is_valid() }
    fn check_constraints(&self) -> ConstraintResult<U, S::Value> {
        self.0.check_constraints()
    }
    fn explain_contradiction(&self) -> Vec<ConstraintViolationDetail> {
        self.0.explain_contradiction()
    }
    fn get_state(&self) -> &S { self.0.get_state() }
}

impl <'a, U, S, St, C> FindFirstSolution<'a, U, S, St, C>
where U: UInt, S: State<U>, St: Strategy<U, S>, C: Constraint<U, S> {
    pub fn new(
        puzzle: &'a mut S,
        strategy: &'a St,
        constraint: &'a mut C,
        force_grid: bool,
    ) -> Self {
        FindFirstSolution(DfsSolver::new(puzzle, strategy, constraint), force_grid)
    }

    pub fn init(&mut self) -> Result<&dyn DfsSolverView<U, S>, Error> {
        self.0.init()?;
        Ok(&self.0)
    }

    pub fn step(&mut self) -> Result<&dyn DfsSolverView<U, S>, Error> {
        self.0.step(self.1)?;
        Ok(&self.0)
    }

    pub fn solve(&mut self) -> Result<Option<&dyn DfsSolverView<U, S>>, Error> {
        self.init()?;
        while !self.0.is_done() {
            self.step()?;
        }
        if self.0.is_valid() {
            return Ok(Some(&self.0));
        } else {
            return Ok(None);
        }
    }

    pub fn solve_debug(&mut self) -> Result<Option<&dyn DfsSolverView<U, S>>, Error> {
        self.init()?;
        while !self.0.is_done() {
            print!("{:?}\n", self.0);
            self.step()?;
        }
        if self.0.is_valid() {
            print!("VALID:\n{:?}\n", self.0);
            return Ok(Some(&self.0));
        } else {
            print!("UNSOLVABLE");
            return Ok(None);
        }
    }
}

/// Find all solutions to the puzzle using the given strategy and constraints.
pub struct FindAllSolutions<'a, U, S, St, C>(DfsSolver<'a, U, S, St, C>, bool)
where U: UInt, S: State<U>, St: Strategy<U, S>, C: Constraint<U, S>;

impl <'a, U, S, St, C> DfsSolverView<U, S>
for FindAllSolutions<'a, U, S, St, C>
where U: UInt, S: State<U>, St: Strategy<U, S>, C: Constraint<U, S> {
    fn solver_state(&self) -> DfsSolverState { self.0.solver_state() }
    fn is_done(&self) -> bool { self.0.solver_state() == DfsSolverState::Exhausted }
    fn is_valid(&self) -> bool { self.0.is_valid() }
    fn check_constraints(&self) -> ConstraintResult<U, S::Value> {
        self.0.check_constraints()
    }
    fn explain_contradiction(&self) -> Vec<ConstraintViolationDetail> {
        self.0.explain_contradiction()
    }
    fn get_state(&self) -> &S { self.0.get_state() }
}

impl <'a, U, S, St, C> FindAllSolutions<'a, U, S, St, C>
where U: UInt, S: State<U>, St: Strategy<U, S>, C: Constraint<U, S> {
    pub fn new(
        puzzle: &'a mut S,
        strategy: &'a St,
        constraint: &'a mut C,
        force_grid: bool,
    ) -> Self {
        FindAllSolutions(DfsSolver::new(puzzle, strategy, constraint), force_grid)
    }

    pub fn init(&mut self) -> Result<&dyn DfsSolverView<U, S>, Error> {
        self.0.init()?;
        Ok(&self.0)
    }

    pub fn step(&mut self) -> Result<&dyn DfsSolverView<U, S>, Error> {
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
    use crate::core::{to_value, Stateful, UVGrid, UVUnwrapped, UVWrapped, UVal, Value};
    use crate::strategy::{CompositeStrategy, PartialStrategy};
    use super::*;

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct GwValue(pub u8);
    impl Value<u8> for GwValue {
        fn parse(_: &str) -> Result<Self, Error> { todo!() }
        fn cardinality() -> usize { 10 }
        fn from_uval(u: UVal<u8, UVUnwrapped>) -> Self { GwValue(u.value()) }
        fn to_uval(self) -> UVal<u8, UVWrapped> { UVal::new(self.0) }
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

        fn get(&self, index: Index) -> Option<Self::Value> {
            if index[0] >= 1 || index[1] >= 8 {
                return None;
            }
            self.digits.get(index).map(to_value)
        }
    }

    impl Stateful<u8, GwValue> for GwLine {
        fn reset(&mut self) {
            self.digits = UVGrid::<u8>::new(Self::ROWS, Self::COLS);
        }

        fn apply(&mut self, index: Index, value: GwValue) -> Result<(), Error> {
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

    #[derive(Debug)]
    struct GwLineConstraint {}
    impl Stateful<u8, GwValue> for GwLineConstraint {}
    impl Constraint<u8, GwLine> for GwLineConstraint {
        fn check(&self, puzzle: &GwLine, _: bool) -> ConstraintResult<u8, GwValue> {
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
                        return ConstraintResult::Contradiction;
                    }
                    let diff: i16 = (i_val as i16) - (j_val as i16);
                    if j == i+1 && diff.abs() < 5 {
                        return ConstraintResult::Contradiction;
                    }
                }
            }
            ConstraintResult::any()
        }
        fn explain_contradictions(&self, _: &GwLine) -> Vec<ConstraintViolationDetail> {
            todo!()
        }
    }

    struct GwLineStrategy {}
    impl Strategy<u8, GwLine> for GwLineStrategy {
        type ActionSet = std::vec::IntoIter<GwValue>;

        fn suggest(&self, puzzle: &GwLine) -> Result<BranchPoint<u8, GwLine, Self::ActionSet>, Error> {
            for i in 0..8 {
                if puzzle.digits.get([0, i]).is_none() {
                    return Ok(BranchPoint::new(
                        [0, i],
                        vec![1, 2, 3, 4, 5, 6, 7, 8, 9].into_iter().map(GwValue).collect::<Vec<_>>().into_iter())
                    );
                }
            }
            Ok(BranchPoint::empty())
        }
    }

    struct GwLineStrategyPartial {}
    impl PartialStrategy<u8, GwLine> for GwLineStrategyPartial {
        fn suggest_partial(&self, puzzle: &GwLine) -> Result<BranchPoint<u8, GwLine, std::vec::IntoIter<GwValue>>, Error> {
            // This is a partial strategy that only affects the first digit and
            // avoids guessing things that can't work.
            if puzzle.digits.get([0, 0]).is_none() {
                return Ok(BranchPoint::new(
                    [0, 0],
                    vec![4, 6].into_iter().map(GwValue).collect::<Vec<_>>().into_iter())
                );
            }
            Ok(BranchPoint::empty())
        }
    }

    #[test]
    fn german_whispers_constraint() {
        let mut puzzle = GwLine::new();
        let constraint = GwLineConstraint {};
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::any());
        puzzle.apply([0, 0], GwValue(1)).unwrap();
        puzzle.apply([0, 3], GwValue(2)).unwrap();
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::any());
        puzzle.apply([0, 5], GwValue(1)).unwrap();
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::Contradiction);
        puzzle.undo([0, 5], GwValue(1)).unwrap();
        puzzle.apply([0, 1], GwValue(3)).unwrap();
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::Contradiction);
        puzzle.undo([0, 1], GwValue(3)).unwrap();
        puzzle.apply([0, 1], GwValue(6)).unwrap();
        let violation = constraint.check(&puzzle, false);
        assert_eq!(violation, ConstraintResult::any());
    }

    #[test]
    fn german_whispers_find() -> Result<(), Error> {
        let mut puzzle = GwLine::new();
        let strategy = GwLineStrategy {};
        let mut constraint = GwLineConstraint {};
        let mut finder = FindFirstSolution::new(&mut puzzle, &strategy, &mut constraint, false);
        let maybe_solution = finder.solve()?;
        assert!(maybe_solution.is_some());
        assert_eq!(maybe_solution.unwrap().get_state().to_string(), "49382716");
        Ok(())
    }

    #[test]
    fn german_whispers_trace() -> Result<(), Error> {
        let mut puzzle = GwLine::new();
        let strategy = GwLineStrategy {};
        let mut constraint = GwLineConstraint {};
        let mut finder = FindFirstSolution::new(&mut puzzle, &strategy, &mut constraint, true);
        let mut steps: usize = 0;
        let mut contradiction_count: usize = 0;
        finder.init().unwrap();
        while !finder.is_done() {
            finder.step()?;
            steps += 1;
            contradiction_count += if finder.check_constraints().no_contradiction() { 0 } else { 1 };
        }
        assert!(finder.is_valid());
        assert!(steps > 100);
        assert!(contradiction_count > 100);
        Ok(())
    }

    #[test]
    fn german_whispers_all() -> Result<(), Error> {
        let mut puzzle = GwLine::new();
        let strategy = GwLineStrategy {};
        let mut constraint = GwLineConstraint {};
        let mut finder = FindAllSolutions::new(&mut puzzle, &strategy, &mut constraint, false);
        let mut steps: usize = 0;
        let mut solution_count: usize = 0;
        finder.init().unwrap();
        while !finder.is_done() {
            finder.step()?;
            steps += 1;
            solution_count += if finder.solver_state() == DfsSolverState::Solved { 1 } else { 0 };
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
        let mut constraint = GwLineConstraint {};
        let mut finder = FindAllSolutions::new(&mut puzzle, &strategy, &mut constraint, false);
        let mut steps: usize = 0;
        let mut solution_count: usize = 0;
        finder.init().unwrap();
        while !finder.is_done() {
            finder.step()?;
            steps += 1;
            solution_count += if finder.solver_state() == DfsSolverState::Solved { 1 } else { 0 };
        }
        assert!(steps < 1000);
        assert_eq!(solution_count, 2);
        Ok(())
    }
}