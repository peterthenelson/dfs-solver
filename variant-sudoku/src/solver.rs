use std::fmt::Debug;
use std::marker::PhantomData;
use crate::core::{Attribution, BranchPoint, ConstraintResult, Error, Index, Key, Overlay, RankingInfo, State, Stateful, Value};
use crate::constraint::Constraint;
use crate::ranker::Ranker;

#[derive(Debug, PartialEq, Clone, Copy, Eq)]
pub struct InitializingState {
    // The index is that of the most recently applied action (during the
    // initial stage when actions already in the grid are replayed).
    last_filled: Option<Index>,
    // The index into the vector of givens for the next given.
    next_given_index: usize,
}

#[derive(Debug, PartialEq, Clone, Copy, Eq)]
pub struct AdvancingState {
    // The number of possibilities at the BranchPoint where this advance was taken.
    pub possibilities: usize,
    // The step at which this advance was taken.
    pub step: usize,
}

#[derive(Debug, PartialEq, Clone, Copy, Eq)]
pub struct BacktrackingState {}

/// The state of the DFS solver. At any point in time, the solver is either
/// advancing (ready to take a new action), backtracking (undoing actions),
/// solved (has found a solution), or exhausted (no more actions to take).
#[derive(Debug, PartialEq, Clone, Copy, Eq)]
pub enum DfsSolverState {
    Initializing(InitializingState),
    Advancing(AdvancingState),
    Backtracking(BacktrackingState),
    InitializationFailed,
    Solved,
    Exhausted,
}

// A view on the state and associated data for the solver.
pub trait DfsSolverView<V, O, R, C>
where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
    fn step_count(&self) -> usize;
    fn solver_state(&self) -> DfsSolverState;
    fn is_initializing(&self) -> bool;
    fn is_done(&self) -> bool;
    fn is_valid(&self) -> bool;
    fn most_recent_action(&self) -> Option<(Index, V)>;
    fn backtracked_steps(&self) -> Option<usize>;
    fn ranker(&self) -> &R;
    fn constraint(&self) -> &C;
    fn constraint_result(&self) -> ConstraintResult<V>;
    fn ranking_info(&self) -> &Option<RankingInfo<V>>;
    fn state(&self) -> &State<V, O>;
}

// Mostly for debugging purposes, a StepObserver allows the caller of various
// solver methods to dump or otherwise inspect the state of the solver after
// each step. This is unlikely to be sufficient to write a fully fledged
// debugger (and certainly not sufficient for a UI), but when debugging failing
// tests, it is much easier to inject a StepObserver than it is to invert
// control and fully instrument the whole solving process.
pub trait StepObserver<V, O, R, C>
where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
    fn after_step(&mut self, solver: &dyn DfsSolverView<V, O, R, C>);
}

pub const MANUAL_ATTRIBUTION: &str = "MANUAL_STEP";

/// DFS solver. If you want a lower-level API that allows for more control over
/// the solving process, you can directly use this. Most users should prefer
/// FindFirstSolution or FindAllSolutions, which are higher-level APIs. However,
/// if you are implementing a UI or debugging, this API may be useful.
pub struct DfsSolver<'a, V, O, R, C>
where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
    step: usize,
    puzzle: &'a mut State<V, O>,
    ranker: &'a R,
    constraint: &'a mut C,
    givens: Vec<(Index, V)>,
    check_result: ConstraintResult<V>,
    ranking_info: Option<RankingInfo<V>>,
    next_decision: Option<BranchPoint<V>>,
    stack: Vec<BranchPoint<V>>,
    backtracked_steps: Option<usize>,
    manual_attr: Key<Attribution>,
    state: DfsSolverState,
    _marker: PhantomData<O>,
}

impl <'a, V, O, R, C> Debug
for DfsSolver<'a, V, O, R, C>
where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State:\n{:?}Constraint:\n{:?}\n", self.puzzle, self.constraint)
    }
}

impl <'a, V, O, R, C> DfsSolverView<V, O, R, C>
for DfsSolver<'a, V, O, R, C>
where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
    fn step_count(&self) -> usize {
        self.step
    }

    fn solver_state(&self) -> DfsSolverState {
        self.state
    }

    fn is_initializing(&self) -> bool {
        if let DfsSolverState::Initializing(_) = self.state {
            true
        } else {
            false
        }
    }

    fn is_done(&self) -> bool {
        match self.state {
            DfsSolverState::InitializationFailed | DfsSolverState::Solved | DfsSolverState::Exhausted => true,
            _ => false,
        }
    }

    fn is_valid(&self) -> bool {
        match self.check_result {
            ConstraintResult::Contradiction(_) => false,
            _ => true,
        }
    }

    fn most_recent_action(&self) -> Option<(Index, V)> {
        if let Some(b) = self.stack.last() {
            b.chosen()
        } else {
            match self.state {
                DfsSolverState::Initializing(InitializingState{ last_filled: Some(index), next_given_index: _ }) => {
                    Some((index, self.puzzle.get(index).unwrap()))
                },
                _ => None,
            }
        }
    }

    fn backtracked_steps(&self) -> Option<usize> { self.backtracked_steps }

    fn constraint(&self) -> &C {
        self.constraint
    }

    fn ranker(&self) -> &R {
        self.ranker
    }

    fn constraint_result(&self) -> ConstraintResult<V> {
        self.check_result.clone()
    }

    fn ranking_info(&self) -> &Option<RankingInfo<V>> {
        &self.ranking_info
    }

    fn state(&self) -> &State<V, O> {
        self.puzzle
    }
}

const NOT_INITIALIZED: Error = Error::new_const("Must call init() before stepping forward");
const PUZZLE_ALREADY_DONE: Error = Error::new_const("Puzzle already done");
const NO_CHOICE: Error = Error::new_const("Decision point has no choice");

impl <'a, V, O, R, C> DfsSolver<'a, V, O, R, C>
where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
    pub fn new(
        puzzle: &'a mut State<V, O>,
        ranker: &'a R,
        constraint: &'a mut C, 
    ) -> Self {
        let givens = puzzle.given_actions();
        DfsSolver {
            step: 0,
            puzzle,
            ranker,
            constraint,
            check_result: ConstraintResult::Ok,
            givens,
            ranking_info: None,
            next_decision: None,
            stack: Vec::new(),
            backtracked_steps: None,
            manual_attr: Key::register(MANUAL_ATTRIBUTION),
            state: DfsSolverState::Initializing(InitializingState { last_filled: None, next_given_index: 0 }),
            _marker: PhantomData,
        }
    }

    fn check_and_rank(&mut self) {
        let mut ranking = self.ranker.init_ranking(&self.puzzle);
        self.check_result = self.constraint.check(self.puzzle, &mut ranking);
        self.ranking_info = None;
        self.next_decision = match &self.check_result {
            ConstraintResult::Contradiction(_) => None,
            ConstraintResult::Certainty(d, a) => {
                Some(BranchPoint::unique(self.step+1, *a, d.index, d.value))
            },
            ConstraintResult::Ok => {
                let (cr, bp) = self.ranker.rank(self.step + 1, &mut ranking, self.puzzle);
                self.check_result = cr;
                self.ranking_info = Some(ranking);
                match &self.check_result {
                    ConstraintResult::Contradiction(_) => None,
                    ConstraintResult::Certainty(d, a) => {
                        Some(BranchPoint::unique(self.step+1, *a, d.index, d.value))
                    },
                    ConstraintResult::Ok => bp,
                }
            },
        };
    }

    fn apply(&mut self, decision: BranchPoint<V>) -> Result<(), Error> {
        if self.is_initializing() {
            return Err(NOT_INITIALIZED);
        } else if self.is_done() {
            return Err(PUZZLE_ALREADY_DONE);
        } else if decision.chosen().is_none() {
            return Err(NO_CHOICE);
        }
        {
            let (i, v) = decision.chosen().unwrap();
            self.puzzle.apply(i, v)?;
            if let Err(e) = self.constraint.apply(i, v) {
                self.puzzle.undo(i, v)?;
                return Err(e);
            }
        }
        let decision_width = decision.remaining() + 1;
        self.stack.push(decision);
        self.check_and_rank();
        self.state = if self.is_valid() {
            DfsSolverState::Advancing(AdvancingState {
                possibilities: decision_width,
                step: self.step,
            })
        } else {
            DfsSolverState::Backtracking(BacktrackingState {})
        };
        return Ok(());
    }

    fn unapply(&mut self, decision: &BranchPoint<V>) -> Result<(), Error> {
        let (i, v) = decision.chosen().unwrap();
        if let Err(e) = self.puzzle.undo(i, v) {
            self.constraint.undo(i, v)?;
            return Err(e);
        }
        self.constraint.undo(i, v)
    }

    /// The stack of BranchPoints
    pub fn stack(&self) -> &Vec<BranchPoint<V>> { &self.stack }

    /// Overriding any logic the solver has, manually do a move.
    pub fn manual_step(&mut self, index: Index, value: V) -> Result<(), Error> {
        self.step += 1;
        self.apply(BranchPoint::unique(self.step, self.manual_attr, index, value))
    }

    /// Force the solver into the backtracking state. (Useful for exhaustively
    /// listing all solutions.)
    pub fn force_backtrack(&mut self) -> bool {
        if self.state == DfsSolverState::Exhausted {
            return false;
        }
        self.step += 1;
        self.state = DfsSolverState::Backtracking(BacktrackingState {});
        true
    }

    /// Undoes the previous action and applies the previous one from the same
    /// stack frame, if any. Unlike force_backtrack(), the solver will
    /// eventually revisit the state before retreat() was called. (Due to the
    /// way backtracking works, it may return immediately or take many steps to
    /// do so.) Returns false if there are no more actions to undo. Note that
    /// the step_count continues to increase.
    pub fn retreat(&mut self) -> Result<bool, Error> {
        self.step += 1;
        if self.stack.is_empty() {
            return Ok(false);
        }
        let mut decision = self.stack.pop().unwrap();
        self.unapply(&decision)?;
        if decision.retreat() {
            self.apply(decision)?;
        } else {
            self.check_and_rank();
            let decision_width = match self.stack.last() {
                Some(d) => d.remaining() + 1,
                None => 0,
            };
            self.state = if self.is_valid() {
                DfsSolverState::Advancing(AdvancingState {
                    possibilities: decision_width,
                    step: self.step,
                })
            } else {
                DfsSolverState::Backtracking(BacktrackingState {})
            };
        }
        Ok(true)
    }

    pub fn step(&mut self) -> Result<(), Error> {
        self.step += 1;
        match self.state {
            DfsSolverState::Initializing(state) => {
                // Make sure that check_and_rank gets called once regardless
                // of whether there are any actual givens to fill in.
                if state.last_filled.is_none() {
                    self.check_and_rank();
                }
                if state.next_given_index < self.givens.len() {
                    let (i, v) = self.givens[state.next_given_index];
                    self.puzzle.apply(i, v)?;
                    if let Err(e) = self.constraint.apply(i, v) {
                        self.puzzle.undo(i, v)?;
                        return Err(e);
                    }
                    self.check_and_rank();
                    self.state = if self.is_valid() {
                        DfsSolverState::Initializing(InitializingState {
                            last_filled: Some(i),
                            next_given_index: state.next_given_index + 1,
                        })
                    } else {
                        DfsSolverState::InitializationFailed
                    };
                } else {
                    self.state = DfsSolverState::Advancing(AdvancingState {
                        possibilities: 0,
                        step: self.step,
                    });
                    if self.ranking_info.is_none() {
                        let mut ranking = self.ranker.init_ranking(&self.puzzle);
                        let _ = self.ranker.rank(self.step, &mut ranking, &self.puzzle);
                        self.ranking_info = Some(ranking);
                    }
                }
                Ok(())
            }
            DfsSolverState::InitializationFailed => Err(PUZZLE_ALREADY_DONE),
            DfsSolverState::Solved => Err(PUZZLE_ALREADY_DONE),
            DfsSolverState::Exhausted => Err(PUZZLE_ALREADY_DONE),
            DfsSolverState::Advancing(_) => {
                // Take a new action
                let decision = self.next_decision.as_ref().unwrap();
                if decision.chosen().is_some() {
                    self.apply(decision.clone())?;
                } else {
                    self.state = DfsSolverState::Solved;
                }
                self.backtracked_steps = None;
                Ok(())
            }
            DfsSolverState::Backtracking(_) => {
                if self.stack.is_empty() {
                    self.state = DfsSolverState::Exhausted;
                    self.backtracked_steps = Some(self.step);
                    return Ok(());
                }
                // Backtrack, attempting to advance an existing action set
                let mut decision = self.stack.pop().unwrap();
                self.backtracked_steps = Some(self.step - decision.branch_step);
                self.unapply(&decision)?;
                match decision.advance() {
                    Some(_) => {
                        self.apply(decision)?;
                        Ok(())
                    }
                    None => {
                        self.state = DfsSolverState::Backtracking(BacktrackingState {});
                        Ok(())
                    },
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.puzzle.reset();
        self.constraint.reset();
        self.check_result = ConstraintResult::Ok;
        self.ranking_info = None;
        self.stack.clear();
        self.state = DfsSolverState::Initializing(InitializingState { last_filled: None, next_given_index: 0 });
        self.step = 0;
        self.backtracked_steps = None;
    }
}

/// Find first solution to the puzzle using the given ranker and constraints.
pub struct FindFirstSolution<'a, V, O, R, C>
where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
    solver: DfsSolver<'a, V, O, R, C>,
    observer: Option<&'a mut dyn StepObserver<V, O, R, C>>,
}

impl <'a, V, O, R, C> DfsSolverView<V, O, R, C>
for FindFirstSolution<'a, V, O, R, C>
where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
    fn step_count(&self) -> usize { self.solver.step_count() }
    fn solver_state(&self) -> DfsSolverState { self.solver.solver_state() }
    fn is_initializing(&self) -> bool { self.solver.is_initializing() }
    fn is_done(&self) -> bool { self.solver.is_done() }
    fn is_valid(&self) -> bool { self.solver.is_valid() }
    fn most_recent_action(&self) -> Option<(Index, V)> {
        self.solver.most_recent_action()
    }
    fn backtracked_steps(&self) -> Option<usize> { self.solver.backtracked_steps() }
    fn ranker(&self) -> &R {
        self.solver.ranker()
    }
    fn constraint(&self) -> &C {
        self.solver.constraint()
    }
    fn constraint_result(&self) -> ConstraintResult<V> {
        self.solver.constraint_result()
    }
    fn ranking_info(&self) -> &Option<RankingInfo<V>> {
        self.solver.ranking_info()
    }
    fn state(&self) -> &State<V, O> { self.solver.state() }
}

impl <'a, V, O, R, C> FindFirstSolution<'a, V, O, R, C>
where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
    pub fn new(
        puzzle: &'a mut State<V, O>,
        ranker: &'a R,
        constraint: &'a mut C,
        observer: Option<&'a mut dyn StepObserver<V, O, R, C>>,
    ) -> Self {
        FindFirstSolution {
            solver: DfsSolver::new(puzzle, ranker, constraint),
            observer,
        }
    }

    pub fn step(&mut self) -> Result<&dyn DfsSolverView<V, O, R, C>, Error> {
        self.solver.step()?;
        Ok(&self.solver)
    }

    pub fn solve(&mut self) -> Result<Option<&dyn DfsSolverView<V, O, R, C>>, Error> {
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
pub struct FindAllSolutions<'a, V, O, R, C>
where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
    solver: DfsSolver<'a, V, O, R, C>,
    observer: Option<&'a mut dyn StepObserver<V, O, R, C>>,
}

impl <'a, V, O, R, C> DfsSolverView<V, O, R, C>
for FindAllSolutions<'a, V, O, R, C>
where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
    fn step_count(&self) -> usize { self.solver.step_count() }
    fn solver_state(&self) -> DfsSolverState { self.solver.solver_state() }
    fn is_initializing(&self) -> bool { self.solver.is_initializing() }
    fn is_done(&self) -> bool { self.solver.solver_state() == DfsSolverState::Exhausted }
    fn is_valid(&self) -> bool { self.solver.is_valid() }
    fn most_recent_action(&self) -> Option<(Index, V)> {
        self.solver.most_recent_action()
    }
    fn backtracked_steps(&self) -> Option<usize> { self.solver.backtracked_steps() }
    fn ranker(&self) -> &R {
        self.solver.ranker()
    }
    fn constraint(&self) -> &C {
        self.solver.constraint()
    }
    fn constraint_result(&self) -> ConstraintResult<V> {
        self.solver.constraint_result()
    }
    fn ranking_info(&self) -> &Option<RankingInfo<V>> {
        self.solver.ranking_info()
    }
    fn state(&self) -> &State<V, O> { self.solver.state() }
}

impl <'a, V, O, R, C> FindAllSolutions<'a, V, O, R, C>
where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
    pub fn new(
        puzzle: &'a mut State<V, O>,
        ranker: &'a R,
        constraint: &'a mut C,
        observer: Option<&'a mut dyn StepObserver<V, O, R, C>>,
    ) -> Self {
        FindAllSolutions {
            solver: DfsSolver::new(puzzle, ranker, constraint), observer,
        }
    }

    pub fn step(&mut self) -> Result<&dyn DfsSolverView<V, O, R, C>, Error> {
        if self.solver.state == DfsSolverState::Solved {
            self.solver.force_backtrack();
        }
        self.solver.step()?;
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

pub trait PuzzleSetter {
    type Value: Value;
    type Overlay: Overlay;
    type Ranker: Ranker<Self::Value, Self::Overlay>;
    type Constraint: Constraint<Self::Value, Self::Overlay>;

    fn name() -> Option<String> { None }
    fn setup() -> (State<Self::Value, Self::Overlay>, Self::Ranker, Self::Constraint);
    // Useful for testing: setup the state with a different set of givens.
    fn setup_with_givens(given: State<Self::Value, Self::Overlay>) -> (State<Self::Value, Self::Overlay>, Self::Ranker, Self::Constraint);
}

#[cfg(any(test, feature = "test-util"))]
pub mod test_util {
    use super::*;

    /// Replayer for a partially or wholly complete puzzle. This is helpful if
    /// you'd like to test a constraint and would prefer to specify the state
    /// after a number of actions, rather than as a sequence of actions.
    pub struct PuzzleReplay<'a, V, O, R, C>
    where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
        solver: DfsSolver<'a, V, O, R, C>,
        observer: Option<&'a mut dyn StepObserver<V, O, R, C>>,
    }

    impl <'a, V, O, R, C> DfsSolverView<V, O, R, C>
    for PuzzleReplay<'a, V, O, R, C>
    where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
        fn step_count(&self) -> usize { self.solver.step_count() }
        fn solver_state(&self) -> DfsSolverState { self.solver.solver_state() }
        fn is_initializing(&self) -> bool { self.solver.is_initializing() }
        fn is_done(&self) -> bool { self.solver.is_done() }
        fn is_valid(&self) -> bool { self.solver.is_valid() }
        fn most_recent_action(&self) -> Option<(Index, V)> {
            self.solver.most_recent_action()
        }
        fn backtracked_steps(&self) -> Option<usize> { self.solver.backtracked_steps() }
        fn ranker(&self) -> &R {
            self.solver.ranker()
        }
        fn constraint(&self) -> &C {
            self.solver.constraint()
        }
        fn constraint_result(&self) -> ConstraintResult<V> {
            self.solver.constraint_result()
        }
        fn ranking_info(&self) -> &Option<RankingInfo<V>> {
            self.solver.ranking_info()
        }
        fn state(&self) -> &State<V, O> { self.solver.state() }
    }

    impl <'a, V, O, R, C> PuzzleReplay<'a, V, O, R, C>
    where V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O> {
        pub fn new(
            puzzle: &'a mut State<V, O>,
            ranker: &'a R,
            constraint: &'a mut C,
            observer: Option<&'a mut dyn StepObserver<V, O, R, C>>,
        ) -> Self {
            Self {
                solver: DfsSolver::new(puzzle, ranker, constraint),
                observer,
            }
        }

        pub fn step(&mut self) -> Result<&dyn DfsSolverView<V, O, R, C>, Error> {
            self.solver.step()?;
            Ok(&self.solver)
        }

        /// Replay all the existing actions in the puzzle against a constraint
        /// and report the final ConstraintResult (or a contradiction is
        /// detected during the replay).
        pub fn replay(&mut self) -> Result<ConstraintResult<V>, Error> {
            while self.solver.is_initializing() {
                self.step()?;
                if let Some(observer) = &mut self.observer {
                    observer.after_step(&self.solver);
                }
                let result = self.solver.constraint_result();
                if let ConstraintResult::Contradiction(_) = result {
                    return Ok(result);
                }
            }
            return Ok(self.solver.constraint_result());
        }
    }

    // In tests, it can be helpful to turn a collection of types into a
    // PuzzleSetter implementation (e.g., to be able to invoke
    // interactive_debug), even if you don't actually need to use any of the
    // actual methods.
    pub struct FakeSetter<V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O>>(PhantomData<(V, O, R, C)>);
    impl <V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O>> FakeSetter<V, O, R, C> {
        pub fn new() -> Self { Self(PhantomData) }
    }
    impl <V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O>>
    PuzzleSetter for FakeSetter<V, O, R, C> {
        type Value = V;
        type Overlay = O;
        type Ranker = R;
        type Constraint = C;
        fn setup() -> (State<V, O>, Self::Ranker, Self::Constraint) {
            panic!("FakeSetter does not implement setup/setup_with_givens and \
                    is only valid to use as a collection of associated types.");
        }
        fn setup_with_givens(_: State<V, O>) -> (State<V, O>, Self::Ranker, Self::Constraint) {
            panic!("FakeSetter does not implement setup/setup_with_givens and \
                    is only valid to use as a collection of associated types.");
        }
    }
}

#[cfg(test)]
mod test {
    use crate::constraint::test_util::assert_contradiction;
    use crate::core::test_util::{OneDimOverlay, TestVal};
    use crate::core::{RankingInfo, Stateful, VSetMut};
    use crate::ranker::StdRanker;
    use super::*;

    type GwOverlay = OneDimOverlay<8>;
    type GwLine = State<TestVal, GwOverlay>;

    #[derive(Debug)]
    struct GwLineConstraint {}
    impl Stateful<TestVal> for GwLineConstraint {}
    impl Constraint<TestVal, GwOverlay> for GwLineConstraint {
        fn check(&self, puzzle: &GwLine, _: &mut RankingInfo<TestVal>) -> ConstraintResult<TestVal> {
            for i in 0..8 {
                if puzzle.get([0, i]).is_none() {
                    continue;
                }
                let i_val = puzzle.get([0, i]).unwrap().0;
                for j in i+1..8 {
                    if puzzle.get([0, j]).is_none() {
                        continue;
                    }
                    let j_val = puzzle.get([0, j]).unwrap().0;
                    if i_val == j_val {
                        return ConstraintResult::Contradiction(Key::register("GW_DUPE"))
                    }
                    let diff: i16 = (i_val as i16) - (j_val as i16);
                    if j == i+1 && diff.abs() < 5 {
                        return ConstraintResult::Contradiction(Key::register("GW_TOO_CLOSE"));
                    }
                }
            }
            ConstraintResult::Ok
        }
        fn debug_at(&self, _: &GwLine, _: Index) -> Option<String> { Some("NA".to_string()) }
    }

    #[derive(Debug)]
    struct GwSmartLineConstraint {}
    impl Stateful<TestVal> for GwSmartLineConstraint {}
    impl Constraint<TestVal, GwOverlay> for GwSmartLineConstraint {
        fn check(&self, puzzle: &GwLine, ranking: &mut RankingInfo<TestVal>) -> ConstraintResult<TestVal> {
            let grid = ranking.cells_mut();
            for i in 0..8 {
                if let Some(v) = puzzle.get([0, i]) {
                    (0..8).for_each(|j| { grid.get_mut([0, j]).0.remove(&v) });
                    (1..=9).for_each(|w| {
                        if (w < v.0 && v.0-w >= 5) || (w > v.0 && w-v.0 >= 5) {
                            return;
                        }
                        if i > 0 {
                            grid.get_mut([0, i-1]).0.remove(&TestVal(w));
                        }
                        if i < 7 {
                            grid.get_mut([0, i+1]).0.remove(&TestVal(w));
                        }
                    });
                }
            }
            ConstraintResult::Ok
        }
        fn debug_at(&self, _: &GwLine, _: Index) -> Option<String> { Some("NA".to_string()) }
    }

    #[test]
    fn test_german_whispers_constraint() {
        let mut puzzle = GwLine::new(GwOverlay {});
        let constraint = GwLineConstraint {};
        let mut ranking = StdRanker::default_negative().init_ranking(&puzzle);
        let violation = constraint.check(&puzzle, &mut ranking);
        assert_eq!(violation, ConstraintResult::Ok);
        puzzle.apply([0, 0], TestVal(1)).unwrap();
        puzzle.apply([0, 3], TestVal(2)).unwrap();
        let violation = constraint.check(&puzzle, &mut ranking);
        assert_eq!(violation, ConstraintResult::Ok);
        puzzle.apply([0, 5], TestVal(1)).unwrap();
        let violation = constraint.check(&puzzle, &mut ranking);
        assert_contradiction(violation, "GW_DUPE");
        puzzle.undo([0, 5], TestVal(1)).unwrap();
        puzzle.apply([0, 1], TestVal(3)).unwrap();
        let violation = constraint.check(&puzzle, &mut ranking);
        assert_contradiction(violation, "GW_TOO_CLOSE");
        puzzle.undo([0, 1], TestVal(3)).unwrap();
        puzzle.apply([0, 1], TestVal(6)).unwrap();
        let violation = constraint.check(&puzzle, &mut ranking);
        assert_eq!(violation, ConstraintResult::Ok);
    }

    struct GwLineSetter;
    impl PuzzleSetter for GwLineSetter {
        type Value = TestVal;
        type Overlay = GwOverlay;
        type Ranker = StdRanker;
        type Constraint = GwLineConstraint;
        fn setup() -> (GwLine, Self::Ranker, Self::Constraint) {
            Self::setup_with_givens(GwLine::new(GwOverlay {}))
        }
        fn setup_with_givens(given: GwLine) -> (GwLine, Self::Ranker, Self::Constraint) {
            (given, StdRanker::default_negative(), GwLineConstraint{})
        }
    }
    struct GwSmartLineSetter;
    impl PuzzleSetter for GwSmartLineSetter {
        type Value = TestVal;
        type Overlay = GwOverlay;
        type Ranker = StdRanker;
        type Constraint = GwSmartLineConstraint;
        fn setup() -> (GwLine, Self::Ranker, Self::Constraint) {
            Self::setup_with_givens(GwLine::new(GwOverlay {}))
        }
        fn setup_with_givens(given: GwLine) -> (GwLine, Self::Ranker, Self::Constraint) {
            (given, StdRanker::default_negative(), GwSmartLineConstraint{})
        }
    }

    #[test]
    fn test_german_whispers_find() -> Result<(), Error> {
        let (mut puzzle, ranker, mut constraint) = GwLineSetter::setup();
        let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, None);
        let maybe_solution = finder.solve()?;
        assert!(maybe_solution.is_some());
        assert_eq!(format!("{:?}", maybe_solution.unwrap().state()), "49382716");
        Ok(())
    }

    #[test]
    fn test_german_whispers_trace_manual() -> Result<(), Error> {
        let (mut puzzle, ranker, mut constraint) = GwLineSetter::setup();
        let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, None);
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
    impl <R: Ranker<TestVal, GwOverlay>, C: Constraint<TestVal, GwOverlay>>
    StepObserver<TestVal, GwOverlay, R, C> for ContraCounter {
        fn after_step(&mut self, solver: &dyn DfsSolverView<TestVal, GwOverlay, R, C>) {
            if !solver.is_valid() {
                self.0 += 1;
            }
        }
    }

    #[test]
    fn test_german_whispers_trace_observer() -> Result<(), Error> {
        let (mut puzzle, ranker, mut constraint) = GwLineSetter::setup();
        let mut counter = ContraCounter(0);
        let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, Some(&mut counter));
        let _ = finder.solve()?;
        assert!(finder.is_valid());
        assert!(counter.0 > 100);
        Ok(())
    }

    #[test]
    fn test_german_whispers_all() -> Result<(), Error> {
        let (mut puzzle, ranker, mut constraint) = GwLineSetter::setup();
        let mut finder = FindAllSolutions::new(&mut puzzle, &ranker, &mut constraint, None);
        let (steps, solution_count) = finder.solve_all()?;
        assert!(steps > 2500);
        assert_eq!(solution_count, 2);
        Ok(())
    }

    #[test]
    fn test_german_whispers_all_fast() -> Result<(), Error> {
        let (mut puzzle, ranker, mut constraint) = GwSmartLineSetter::setup();
        let mut finder = FindAllSolutions::new(&mut puzzle, &ranker, &mut constraint, None);
        let (steps, solution_count) = finder.solve_all()?;
        assert!(steps < 500);
        assert_eq!(solution_count, 2);
        Ok(())
    }

    #[test]
    fn test_german_whispers_undo_works() -> Result<(), Error> {
        // First runthrough to collect the moves in order.
        let expected_solution = {
            let (mut puzzle, ranker, mut constraint) = GwSmartLineSetter::setup();
            let mut solver = DfsSolver::new(&mut puzzle, &ranker, &mut constraint);
            while !solver.is_done() {
                solver.step()?;
            }
            assert!(solver.is_valid());
            (0..puzzle.overlay().grid_dims().1).map(|i| puzzle.get([0, i])).collect::<Vec<_>>()
        };
        // Next runthrough does undo every once in a while.
        let actual_solution = {
            let (mut puzzle, ranker, mut constraint) = GwSmartLineSetter::setup();
            let mut i = 1;
            let mut solver = DfsSolver::new(&mut puzzle, &ranker, &mut constraint);
            while !solver.is_done() {
                if i % 23 == 0 {
                    solver.retreat()?;
                } else {
                    solver.step()?;
                }
                i += 1;
            }
            (0..puzzle.overlay().grid_dims().1).map(|i| puzzle.get([0, i])).collect::<Vec<_>>()
        };
        assert_eq!(actual_solution, expected_solution);
        Ok(())
    }
}