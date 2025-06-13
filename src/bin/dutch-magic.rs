use std::time::Duration;
use variant_sudoku_dfs::core::FeatureVec;
use variant_sudoku_dfs::dutch_whispers::{DutchWhisperBuilder, DutchWhisperChecker};
use variant_sudoku_dfs::magic_squares::{MagicSquare, MagicSquareChecker, MS_FEATURE};
use variant_sudoku_dfs::ranker::{OverlaySensitiveLinearRanker, NUM_POSSIBLE_FEATURE};
use variant_sudoku_dfs::constraint::MultiConstraint;
use variant_sudoku_dfs::solver::{FindFirstSolution, StepObserver};
use variant_sudoku_dfs::debug::{DbgObserver, Sample};
use variant_sudoku_dfs::sudoku::{nine_standard_overlay, SState, StandardSudokuChecker, StandardSudokuOverlay};
use variant_sudoku_dfs::cages::{CageBuilder, CageChecker, CAGE_FEATURE};

type NineStd = SState<9, 9, 1, 9, StandardSudokuOverlay<9, 9>>;

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000NRF
fn solve<D: StepObserver<u8, NineStd>>(
    given: Option<NineStd>,
    mut observer: D,
) {
    // No given digits in real puzzle but can be passed in in test.
    let mut puzzle = given.unwrap_or(
        SState::<9, 9, 1, 9, _>::new(nine_standard_overlay())
    );
    let cb = CageBuilder::new(false, puzzle.get_overlay());
    let cages = vec![
        cb.v([2, 4], [3, 4]),
        cb.v([4, 5], [4, 6]),
    ];
    let dw = DutchWhisperBuilder::new(puzzle.get_overlay());
    let whispers = vec![
        dw.row([0, 0], 3),
        dw.row([1, 4], 3),
        dw.row([3, 0], 3),
        dw.row([4, 3], 3),
        dw.row([5, 6], 3),
        dw.row([7, 2], 3),
        dw.row([8, 6], 3),
    ];
    let squares = vec![
        MagicSquare::new([1, 5]),
        MagicSquare::new([3, 1]),
        MagicSquare::new([5, 7]),
        MagicSquare::new([7, 3]),
    ];
    let mut constraint = MultiConstraint::new(vec_box::vec_box![
        StandardSudokuChecker::new(&puzzle),
        CageChecker::new(cages),
        DutchWhisperChecker::new(whispers),
        MagicSquareChecker::new(squares),
    ]);
    let ranker = OverlaySensitiveLinearRanker::new(FeatureVec::from_pairs(vec![
        (NUM_POSSIBLE_FEATURE, -100.0),
        (CAGE_FEATURE, 1.0),
        (MS_FEATURE, 1.0),
    ]), |_, x, y| x+y);
    let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, Some(&mut observer));
    let maybe_solution = finder.solve().expect("Puzzle solver returned an error:");
    println!("Solution:\n{}", maybe_solution.expect("No solution found!").get_state().serialize());
}

pub fn main() {
    let mut dbg = DbgObserver::new();
    dbg.sample_print(Sample::every_n(1000))
        .sample_stats("figures/dutch-magic.png", Sample::time(Duration::from_secs(30)));
    solve(None, dbg);
}

#[cfg(test)]
mod test {
    use variant_sudoku_dfs::{debug::NullObserver, sudoku::nine_standard_parse};
    use super::*;

    #[test]
    fn test_dutch_magic_solution() {
        let input: &str = "195643827\n\
                           627895143\n\
                           438127695\n\
                           951438762\n\
                           276951438\n\
                           843276951\n\
                           514389276\n\
                           769512384\n\
                           382764519\n";
        let sudoku = nine_standard_parse(input).unwrap();
        let obs = NullObserver;
        solve(Some(sudoku), obs);
    }
}