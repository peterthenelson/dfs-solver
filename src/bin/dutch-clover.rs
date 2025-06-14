use std::time::Duration;
use variant_sudoku_dfs::core::FeatureVec;
use variant_sudoku_dfs::dutch_whispers::{DutchWhisperBuilder, DutchWhisperChecker, DW_FEATURE};
use variant_sudoku_dfs::ranker::{OverlaySensitiveLinearRanker, NUM_POSSIBLE_FEATURE};
use variant_sudoku_dfs::constraint::MultiConstraint;
use variant_sudoku_dfs::solver::{FindFirstSolution, StepObserver};
use variant_sudoku_dfs::debug::{DbgObserver, Sample};
use variant_sudoku_dfs::sudoku::{nine_standard_parse, SState, StandardSudokuChecker, StandardSudokuOverlay};

type NineStd = SState<9, 9, 1, 9, StandardSudokuOverlay<9, 9>>;

// https://sudokupad.app/clover/dec-1-2023-dutch-whispers
fn solve<D: StepObserver<u8, NineStd>>(
    given: Option<NineStd>,
    mut observer: D,
) {
    // The given digits in real puzzle but can be overridden in in test.
    let mut puzzle = given.unwrap_or(
        nine_standard_parse(
            "..5.6.7..\n\
             .........\n\
             ....3.4.5\n\
             .........\n\
             .........\n\
             .........\n\
             2.3.4....\n\
             .........\n\
             ..6.7.8..\n"
        ).unwrap()
    );
    let dw = DutchWhisperBuilder::new(puzzle.get_overlay());
    let whispers = vec![
        dw.whisper(vec![
            [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8],
            [1, 8], [1, 7], [1, 6], [1, 5], [1, 4], [1, 3], [1, 2], [1, 1], [1, 0],
            [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8],
        ]),
        dw.whisper(vec![
            [3, 8], [3, 7], [3, 6], [3, 5], [3, 4], [3, 3], [3, 2], [3, 1], [3, 0],
            [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
            [5, 8], [5, 7], [5, 6], [5, 5], [5, 4], [5, 3], [5, 2], [5, 1], [5, 0],
        ]),
        dw.whisper(vec![
            [6, 8], [6, 7], [6, 6], [6, 5], [6, 4], [6, 3], [6, 2], [6, 1], [6, 0],
            [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8],
            [8, 8], [8, 7], [8, 6], [8, 5], [8, 4], [8, 3], [8, 2], [8, 1], [8, 0],
        ]),
    ];
    let mut constraint = MultiConstraint::new(vec_box::vec_box![
        StandardSudokuChecker::new(&puzzle),
        DutchWhisperChecker::new(whispers),
    ]);
    let ranker = OverlaySensitiveLinearRanker::new(FeatureVec::from_pairs(vec![
        (NUM_POSSIBLE_FEATURE, -100.0),
        (DW_FEATURE, 1.0)
    ]), |_, x, y| x+y);
    let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, Some(&mut observer));
    let maybe_solution = finder.solve().expect("Puzzle solver returned an error:");
    println!("Solution:\n{}", maybe_solution.expect("No solution found!").get_state().serialize());
}

pub fn main() {
    let mut dbg = DbgObserver::new();
    dbg.sample_print(Sample::every_n(100))
        .sample_stats("figures/dutch-clover.png", Sample::time(Duration::from_secs(30)));
    solve(None, dbg);
}

#[cfg(test)]
mod test {
    use variant_sudoku_dfs::{debug::NullObserver, sudoku::nine_standard_parse};
    use super::*;

    #[test]
    fn test_dutch_clover_solution() {
        let input: &str = "495162738\n\
                           738495162\n\
                           162738495\n\
                           951627384\n\
                           384951627\n\
                           627384951\n\
                           273849516\n\
                           849516273\n\
                           51627384.\n";
        let sudoku = nine_standard_parse(input).unwrap();
        let obs = NullObserver;
        solve(Some(sudoku), obs);
    }
}