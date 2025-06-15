use crate::{constraint::MultiConstraint, ranker::Ranker, solver::{FindFirstSolution, PuzzleSetter, StepObserver}, sudoku::NineStd};

// TODO: Currently just pulling stuff out of the binaries, but eventually pull
// all the ratatui stuff in here.
pub fn cli_solve<D: StepObserver<u8, NineStd>, R: Ranker<u8, NineStd>, P: PuzzleSetter<u8, NineStd, R, MultiConstraint<u8, NineStd>>>(
    given: Option<NineStd>,
    mut observer: D,
) {
    let (mut s, r, mut c) = if let Some(given) = given {
        P::setup_with_givens(given)
    } else {
        P::setup()
    };
    let mut finder = FindFirstSolution::new(&mut s, &r, &mut c, Some(&mut observer));
    let maybe_solution = finder.solve().expect("Puzzle solver returned an error:");
    println!("Solution:\n{}", maybe_solution.expect("No solution found!").get_state().serialize());
}
