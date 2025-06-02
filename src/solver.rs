use std::fmt::Debug;

use crate::core::{unpack_values, BranchPoint, DecisionGrid, Error, GridIndex, Index, State, UInt};
use crate::constraint::{Constraint, ConstraintResult, ConstraintViolationDetail, Possibilities};
use crate::ranker::Ranker;

#[derive(Debug, PartialEq, Clone, Copy, Eq)]
pub struct AdvancingState {
    pub streak: usize,
    pub possibilities: usize,
    // TODO: pub step: usize,
}

/// The state of the DFS solver. At any point in time, the solver is either
/// advancing (ready to take a new action), backtracking (undoing actions),
/// solved (has found a solution), or exhausted (no more actions to take).
#[derive(Debug, PartialEq, Clone, Copy, Eq)]
pub enum DfsSolverState {
    // The index is that of the most recently applied action (during the
    // initial stage when actions already in the grid are replayed).
    Init(Option<Index>),
    // The values here are the number of actions taken since the advancing stage
    // was entered and the size of the possibility set.
    Advancing(AdvancingState),
    // The value here is the length of the backtrack (so far).
    Backtracking(usize),
    Solved,
    Exhausted,
}

// A view on the state and associated data for the solver.
pub trait DfsSolverView<U: UInt, S: State<U>> {
    fn solver_state(&self) -> DfsSolverState;
    fn is_initializing(&self) -> bool;
    fn is_done(&self) -> bool;
    fn is_valid(&self) -> bool;
    fn most_recent_action(&self) -> Option<(Index, S::Value)>;
    fn get_constraint(&self) -> &dyn Constraint<U, S>;
    fn constraint_result(&self) -> ConstraintResult<U, S::Value>;
    fn explain_contradiction(&self) -> Vec<ConstraintViolationDetail>;
    fn get_state(&self) -> &S;
}

// Mostly for debugging purposes, a StepObserver allows the caller of various
// solver methods to dump or otherwise inspect the state of the solver after
// each step. This is unlikely to be sufficient to write a fully fledged
// debugger (and certainly not sufficient for a UI), but when debugging failing
// tests, it is much easier to inject a StepObserver than it is to invert
// control and fully instrument the whole solving process.
pub trait StepObserver<U: UInt, S: State<U>> {
    fn after_step(&mut self, solver: &dyn DfsSolverView<U, S>);
}

/// DFS solver. If you want a lower-level API that allows for more control over
/// the solving process, you can directly use this. Most users should prefer
/// FindFirstSolution or FindAllSolutions, which are higher-level APIs. However,
/// if you are implementing a UI or debugging, this API may be useful.
pub struct DfsSolver<'a, U, S, R, C>
where U: UInt, S: State<U>, R: Ranker<U, S>, C: Constraint<U, S> {
    puzzle: &'a mut S,
    ranker: &'a R,
    constraint: &'a mut C,
    check_result: ConstraintResult<U, S::Value>,
    stack: Vec<BranchPoint<U, S>>,
    state: DfsSolverState,
}

impl <'a, U, S, R, C> Debug
for DfsSolver<'a, U, S, R, C>
where U: UInt, S: State<U>, R: Ranker<U, S>, C: Constraint<U, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State:\n{:?}Constraint:\n{:?}\n", self.puzzle, self.constraint)
    }
}

impl <'a, U, S, R, C> DfsSolverView<U, S>
for DfsSolver<'a, U, S, R, C>
where U: UInt, S: State<U>, R: Ranker<U, S>, C: Constraint<U, S> {
    fn solver_state(&self) -> DfsSolverState {
        self.state
    }

    fn is_initializing(&self) -> bool {
        if let DfsSolverState::Init(_) = self.state {
            true
        } else {
            false
        }
    }

    fn is_done(&self) -> bool {
        self.state == DfsSolverState::Solved || self.state == DfsSolverState::Exhausted
    }

    fn is_valid(&self) -> bool {
        !self.check_result.has_contradiction(self.puzzle)
    }

    fn most_recent_action(&self) -> Option<(Index, S::Value)> {
        if let Some(b) = self.stack.last() {
            Some((b.index, b.chosen.unwrap()))
        } else {
            match self.state {
                DfsSolverState::Init(Some(index)) => {
                    Some((index, self.puzzle.get(index).unwrap()))
                },
                _ => None,
            }
        }
    }

    fn get_constraint(&self) -> &dyn Constraint<U, S> {
        self.constraint
    }

    fn constraint_result(&self) -> ConstraintResult<U, S::Value> {
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

fn next_filled<U: UInt, S: State<U>>(index: Option<Index>, puzzle: &S) -> Option<(Index, S::Value)> {
    let mut i = if index.is_none() {
        [0, 0]
    } else {
        let mut i2 = index.unwrap();
        i2.increment(S::ROWS, S::COLS);
        i2
    };
    while i.in_bounds(S::ROWS, S::COLS) {
        let v = puzzle.get(i);
        if v.is_some() {
            return Some((i, v.unwrap()));
        }
        i.increment(S::ROWS, S::COLS);
    }
    None
}
 
impl <'a, U, S, R, C> DfsSolver<'a, U, S, R, C>
where U: UInt, S: State<U>, R: Ranker<U, S>, C: Constraint<U, S> {
    pub fn new(
        puzzle: &'a mut S,
        ranker: &'a R,
        constraint: &'a mut C, 
    ) -> Self {
        DfsSolver {
            puzzle,
            ranker,
            constraint,
            check_result: ConstraintResult::any(),
            stack: Vec::new(),
            state: DfsSolverState::Init(None),
        }
    }

    fn apply(&mut self, decision: BranchPoint<U, S>, force_grid: bool) -> Result<(), Error> {
        if self.is_initializing() {
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
        let decision_width = decision.alternatives.len() + 1;
        self.stack.push(decision);
        self.check_result = self.constraint.check(self.puzzle, force_grid);
        self.state = if self.is_valid() {
            DfsSolverState::Advancing(AdvancingState {
                possibilities: decision_width,
                streak: match self.state {
                    DfsSolverState::Advancing(adv) => adv.streak + 1,
                    _ => 1,
                },
            })
        } else {
            DfsSolverState::Backtracking(1)
        };
        return Ok(());
    }

    fn undo(&mut self, decision: &BranchPoint<U, S>) -> Result<(), Error> {
        let v = decision.chosen.unwrap();
        if let Err(e) = self.puzzle.undo(decision.index, v) {
            self.constraint.undo(decision.index, v)?;
            return Err(e);
        }
        self.constraint.undo(decision.index, v)
    }

    fn suggest(&self) -> BranchPoint<U, S> {
        if let Some(d) = self.check_result.has_certainty(self.puzzle) {
            return BranchPoint::unique(d.index, d.value);
        }
        let g: DecisionGrid<U, S::Value> = match &self.check_result {
            ConstraintResult::Other(Possibilities::Any) => DecisionGrid::full(S::ROWS, S::COLS),
            ConstraintResult::Other(Possibilities::Grid(g)) => g.clone(),
            _ => panic!("Unexpected check_result: {:?}", self.check_result),
        };
        if let Some(i) = self.ranker.top(&g, self.puzzle) {
            BranchPoint::new(i, unpack_values(&g.get(i).0))
        } else {
            BranchPoint::empty()
        }
    }

    pub fn manual_step(&mut self, index: Index, value: S::Value, force_grid: bool) -> Result<(), Error> {
        self.apply(BranchPoint::unique(index, value), force_grid)
    }

    pub fn force_backtrack(&mut self) -> bool {
        if self.state == DfsSolverState::Exhausted {
            return false;
        }
        self.state = DfsSolverState::Backtracking(1);
        true
    }

    pub fn step(&mut self, force_grid: bool) -> Result<(), Error> {
        match self.state {
            DfsSolverState::Init(index) => {
                if let Some((i, v)) = next_filled(index, self.puzzle) {
                    self.constraint.apply(i, v)?;
                    self.check_result = self.constraint.check(self.puzzle, force_grid);
                    self.state = DfsSolverState::Init(Some(i));
                } else {
                    self.state = DfsSolverState::Advancing(AdvancingState {
                        streak: 0,
                        possibilities: 0,
                    });
                }
                Ok(())
            }
            DfsSolverState::Solved => Err(PUZZLE_ALREADY_DONE),
            DfsSolverState::Exhausted => Err(PUZZLE_ALREADY_DONE),
            DfsSolverState::Advancing(_) => {
                // Take a new action
                let decision = self.suggest();
                if decision.chosen.is_some() {
                    self.apply(decision, force_grid)?;
                } else {
                    self.state = DfsSolverState::Solved;
                }
                Ok(())
            }
            DfsSolverState::Backtracking(n) => {
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
                    None => {
                        self.state = DfsSolverState::Backtracking(n+1);
                        Ok(())
                    },
                }
            }
        }
    }

    // TODO: This actually resets the puzzle, including any initial moves. Oops.
    pub fn reset(&mut self) {
        self.puzzle.reset();
        self.check_result = ConstraintResult::any();
        self.stack.clear();
        self.state = DfsSolverState::Advancing(AdvancingState {
            streak: 0,
            possibilities: 0,
        });
    }
}

/// Find first solution to the puzzle using the given ranker and constraints.
pub struct FindFirstSolution<'a, U, S, R, C>
where U: UInt, S: State<U>, R: Ranker<U, S>, C: Constraint<U, S> {
    solver: DfsSolver<'a, U, S, R, C>,
    force_grid: bool,
    observer: Option<&'a mut dyn StepObserver<U, S>>,
}

impl <'a, U, S, R, C> DfsSolverView<U, S>
for FindFirstSolution<'a, U, S, R, C>
where U: UInt, S: State<U>, R: Ranker<U, S>, C: Constraint<U, S> {
    fn solver_state(&self) -> DfsSolverState { self.solver.solver_state() }
    fn is_initializing(&self) -> bool { self.solver.is_initializing() }
    fn is_done(&self) -> bool { self.solver.is_done() }
    fn is_valid(&self) -> bool { self.solver.is_valid() }
    fn most_recent_action(&self) -> Option<(Index, S::Value)> {
        self.solver.most_recent_action()
    }
    fn get_constraint(&self) -> &dyn Constraint<U, S> {
        self.solver.get_constraint()
    }
    fn constraint_result(&self) -> ConstraintResult<U, S::Value> {
        self.solver.constraint_result()
    }
    fn explain_contradiction(&self) -> Vec<ConstraintViolationDetail> {
        self.solver.explain_contradiction()
    }
    fn get_state(&self) -> &S { self.solver.get_state() }
}

impl <'a, U, S, R, C> FindFirstSolution<'a, U, S, R, C>
where U: UInt, S: State<U>, R: Ranker<U, S>, C: Constraint<U, S> {
    pub fn new(
        puzzle: &'a mut S,
        ranker: &'a R,
        constraint: &'a mut C,
        force_grid: bool,
        observer: Option<&'a mut dyn StepObserver<U, S>>,
    ) -> Self {
        FindFirstSolution {
            solver: DfsSolver::new(puzzle, ranker, constraint),
            force_grid,
            observer,
        }
    }

    pub fn step(&mut self) -> Result<&dyn DfsSolverView<U, S>, Error> {
        self.solver.step(self.force_grid)?;
        Ok(&self.solver)
    }

    pub fn solve(&mut self) -> Result<Option<&dyn DfsSolverView<U, S>>, Error> {
        while !self.solver.is_done() {
            self.step()?;
            if let Some(observer) = &mut self.observer {
                observer.after_step(&self.solver);
            }
        }
        if self.solver.is_valid() {
            return Ok(Some(&self.solver));
        } else {
            return Ok(None);
        }
    }
}

/// Find all solutions to the puzzle using the given ranker and constraints.
pub struct FindAllSolutions<'a, U, S, R, C>
where U: UInt, S: State<U>, R: Ranker<U, S>, C: Constraint<U, S> {
    solver: DfsSolver<'a, U, S, R, C>,
    force_grid: bool,
    observer: Option<&'a mut dyn StepObserver<U, S>>,
}

impl <'a, U, S, R, C> DfsSolverView<U, S>
for FindAllSolutions<'a, U, S, R, C>
where U: UInt, S: State<U>, R: Ranker<U, S>, C: Constraint<U, S> {
    fn solver_state(&self) -> DfsSolverState { self.solver.solver_state() }
    fn is_initializing(&self) -> bool { self.solver.is_initializing() }
    fn is_done(&self) -> bool { self.solver.solver_state() == DfsSolverState::Exhausted }
    fn is_valid(&self) -> bool { self.solver.is_valid() }
    fn most_recent_action(&self) -> Option<(Index, S::Value)> {
        self.solver.most_recent_action()
    }
    fn get_constraint(&self) -> &dyn Constraint<U, S> {
        self.solver.get_constraint()
    }
    fn constraint_result(&self) -> ConstraintResult<U, S::Value> {
        self.solver.constraint_result()
    }
    fn explain_contradiction(&self) -> Vec<ConstraintViolationDetail> {
        self.solver.explain_contradiction()
    }
    fn get_state(&self) -> &S { self.solver.get_state() }
}

impl <'a, U, S, R, C> FindAllSolutions<'a, U, S, R, C>
where U: UInt, S: State<U>, R: Ranker<U, S>, C: Constraint<U, S> {
    pub fn new(
        puzzle: &'a mut S,
        ranker: &'a R,
        constraint: &'a mut C,
        force_grid: bool,
        observer: Option<&'a mut dyn StepObserver<U, S>>,
    ) -> Self {
        FindAllSolutions {
            solver: DfsSolver::new(puzzle, ranker, constraint),
            force_grid,
            observer,
        }
    }

    pub fn step(&mut self) -> Result<&dyn DfsSolverView<U, S>, Error> {
        if self.solver.state == DfsSolverState::Solved {
            self.solver.force_backtrack();
        }
        self.solver.step(self.force_grid)?;
        Ok(&self.solver)
    }

    // Returns the number of steps taken and the number of solutions found.
    // Obviously unless you provided a StepObserver, you won't be able to see
    // any of the solutions. Prefer directly hitting step() yourself if that
    // is your use-case.
    pub fn solve_all(&mut self) -> Result<(usize, usize), Error> {
        let mut steps = 0;
        let mut solution_count = 0;
        while !self.is_done() {
            self.step()?;
            steps += 1;
            solution_count += if self.solver_state() == DfsSolverState::Solved { 1 } else { 0 };
            if let Some(observer) = &mut self.observer {
                observer.after_step(&self.solver);
            }
        }
        Ok((steps, solution_count))
    }
}

#[cfg(test)]
pub mod test_util {
    use super::*;

    /// Replayer for a partially or wholly complete puzzle. This is helpful if
    /// you'd like to test a constraint and would prefer to specify the state
    /// after a number of actions, rather than as a sequence of actions.
    pub struct PuzzleReplay<'a, U, S, R, C>
    where U: UInt, S: State<U>, R: Ranker<U, S>, C: Constraint<U, S> {
        solver: DfsSolver<'a, U, S, R, C>,
        force_grid: bool,
        observer: Option<&'a mut dyn StepObserver<U, S>>,
    }

    impl <'a, U, S, R, C> DfsSolverView<U, S>
    for PuzzleReplay<'a, U, S, R, C>
    where U: UInt, S: State<U>, R: Ranker<U, S>, C: Constraint<U, S> {
        fn solver_state(&self) -> DfsSolverState { self.solver.solver_state() }
        fn is_initializing(&self) -> bool { self.solver.is_initializing() }
        fn is_done(&self) -> bool { self.solver.is_done() }
        fn is_valid(&self) -> bool { self.solver.is_valid() }
        fn most_recent_action(&self) -> Option<(Index, S::Value)> {
            self.solver.most_recent_action()
        }
        fn get_constraint(&self) -> &dyn Constraint<U, S> {
            self.solver.get_constraint()
        }
        fn constraint_result(&self) -> ConstraintResult<U, S::Value> {
            self.solver.constraint_result()
        }
        fn explain_contradiction(&self) -> Vec<ConstraintViolationDetail> {
            self.solver.explain_contradiction()
        }
        fn get_state(&self) -> &S { self.solver.get_state() }
    }

    impl <'a, U, S, R, C> PuzzleReplay<'a, U, S, R, C>
    where U: UInt, S: State<U>, R: Ranker<U, S>, C: Constraint<U, S> {
        pub fn new(
            puzzle: &'a mut S,
            ranker: &'a R,
            constraint: &'a mut C,
            force_grid: bool,
            observer: Option<&'a mut dyn StepObserver<U, S>>,
        ) -> Self {
            Self {
                solver: DfsSolver::new(puzzle, ranker, constraint),
                force_grid,
                observer,
            }
        }

        pub fn step(&mut self) -> Result<&dyn DfsSolverView<U, S>, Error> {
            self.solver.step(self.force_grid)?;
            Ok(&self.solver)
        }

        /// Replay all the existing actions in the puzzle against a constraint
        /// and report the final ConstraintResult (or a contradiction is
        /// detected during the replay).
        pub fn replay(&mut self) -> Result<ConstraintResult<U, S::Value>, Error> {
            while self.solver.is_initializing() {
                self.step()?;
                if let Some(observer) = &mut self.observer {
                    observer.after_step(&self.solver);
                }
                let result = self.solver.constraint_result();
                if result.has_contradiction(self.solver.get_state()) {
                    return Ok(result);
                }
            }
            return Ok(self.solver.constraint_result());
        }
    }

    // Assertion for a contradiction or lack-thereof
    pub fn assert_contradiction_eq<U: UInt, S: State<U>>(
        constraint: &dyn Constraint<U, S>,
        puzzle: &S,
        result: &ConstraintResult<U, S::Value>,
        expected_contradiction: bool,
    ) {
        let actual = result.has_contradiction(puzzle);
        if expected_contradiction && !actual {
            panic!("Expected contradiction; none found:\nPuzzle state:\n{:?}\n{:?}\nResult: {:?}\n", puzzle, constraint, result);
        } else if actual && !expected_contradiction {
            panic!("Expected no contradiction; one found:\nPuzzle state:\n{:?}\n{:?}\nResult: {:?}\n", puzzle, constraint, result);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::constraint::ConstraintViolationDetail;
    use crate::core::{to_value, Stateful, UVGrid, UVUnwrapped, UVWrapped, UVal, Value};
    use crate::ranker::LinearRanker;
    use super::*;

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct GwValue(pub u8);
    impl Value<u8> for GwValue {
        fn parse(_: &str) -> Result<Self, Error> { todo!() }
        fn cardinality() -> usize { 9 }
        fn possiblities() -> Vec<Self> { (1..10).map(GwValue).collect() }
        fn from_uval(u: UVal<u8, UVUnwrapped>) -> Self { GwValue(u.value()+1) }
        fn to_uval(self) -> UVal<u8, UVWrapped> { UVal::new(self.0-1) }
    }

    #[derive(Clone)]
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

    impl std::fmt::Debug for GwLine {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}\n", self.to_string())
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

    #[derive(Debug)]
    struct GwSmartLineConstraint {}
    impl Stateful<u8, GwValue> for GwSmartLineConstraint {}
    impl Constraint<u8, GwLine> for GwSmartLineConstraint {
        fn check(&self, puzzle: &GwLine, _: bool) -> ConstraintResult<u8, GwValue> {
            let mut grid = DecisionGrid::<u8, GwValue>::full(1, 8);
            for i in 0..8 {
                if let Some(u) = puzzle.digits.get([0, i]) {
                    let v = to_value::<u8, GwValue>(u);
                    (0..8).for_each(|j| { grid.get_mut([0, j]).0.remove(u) });
                    (1..=9).for_each(|w| {
                        if (w < v.0 && v.0-w >= 5) || (w > v.0 && w-v.0 >= 5) {
                            return;
                        }
                        if i > 0 {
                            grid.get_mut([0, i-1]).0.remove(GwValue(w).to_uval());
                        }
                        if i < 7 {
                            grid.get_mut([0, i+1]).0.remove(GwValue(w).to_uval());
                        }
                    });
                }
            }
            ConstraintResult::grid(grid)
        }
        fn explain_contradictions(&self, _: &GwLine) -> Vec<ConstraintViolationDetail> {
            todo!()
        }
    }

    #[test]
    fn test_german_whispers_constraint() {
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
    fn test_german_whispers_find() -> Result<(), Error> {
        let mut puzzle = GwLine::new();
        let ranker = LinearRanker::default();
        let mut constraint = GwLineConstraint {};
        let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, false, None);
        let maybe_solution = finder.solve()?;
        assert!(maybe_solution.is_some());
        assert_eq!(maybe_solution.unwrap().get_state().to_string(), "49382716");
        Ok(())
    }

    #[test]
    fn test_german_whispers_trace_manual() -> Result<(), Error> {
        let mut puzzle = GwLine::new();
        let ranker = LinearRanker::default();
        let mut constraint = GwLineConstraint {};
        let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, true, None);
        let mut steps: usize = 0;
        let mut contradiction_count: usize = 0;
        while !finder.is_done() {
            finder.step()?;
            steps += 1;
            contradiction_count += if finder.is_valid() { 0 } else { 1 };
        }
        assert!(finder.is_valid());
        assert!(steps > 100);
        assert!(contradiction_count > 100);
        Ok(())
    }

    struct ContraCounter(pub usize);
    impl StepObserver<u8, GwLine> for ContraCounter {
        fn after_step(&mut self, solver: &dyn DfsSolverView<u8, GwLine>) {
            if !solver.is_valid() {
                self.0 += 1;
            }
        }
    }

    #[test]
    fn test_german_whispers_trace_observer() -> Result<(), Error> {
        let mut puzzle = GwLine::new();
        let ranker = LinearRanker::default();
        let mut constraint = GwLineConstraint {};
        let mut counter = ContraCounter(0);
        let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, true, Some(&mut counter));
        let _ = finder.solve()?;
        assert!(finder.is_valid());
        assert!(counter.0 > 100);
        Ok(())
    }

    #[test]
    fn test_german_whispers_all() -> Result<(), Error> {
        let mut puzzle = GwLine::new();
        let ranker = LinearRanker::default();
        let mut constraint = GwLineConstraint {};
        let mut finder = FindAllSolutions::new(&mut puzzle, &ranker, &mut constraint, false, None);
        let (steps, solution_count) = finder.solve_all()?;
        assert!(steps > 2500);
        assert_eq!(solution_count, 2);
        Ok(())
    }

    #[test]
    fn test_german_whispers_all_fast() -> Result<(), Error> {
        let mut puzzle = GwLine::new();
        let ranker = LinearRanker::default();
        let mut constraint = GwSmartLineConstraint {};
        let mut finder = FindAllSolutions::new(&mut puzzle, &ranker, &mut constraint, false, None);
        let (steps, solution_count) = finder.solve_all()?;
        assert!(steps < 500);
        assert_eq!(solution_count, 2);
        Ok(())
    }
}