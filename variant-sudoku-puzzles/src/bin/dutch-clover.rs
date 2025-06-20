use variant_sudoku::core::FeatureVec;
use variant_sudoku::dutch_whispers::{DutchWhisperBuilder, DutchWhisperChecker, DW_FEATURE};
use variant_sudoku::ranker::{StdRanker, NUM_POSSIBLE_FEATURE};
use variant_sudoku::constraint::MultiConstraint;
use variant_sudoku::solver::PuzzleSetter;
use variant_sudoku::sudoku::{nine_standard_parse, NineStd, NineStdOverlay, NineStdVal, StdChecker};
use variant_sudoku::tui::solve_main;
use variant_sudoku::tui_std::NineStdTui;

// https://sudokupad.app/clover/dec-1-2023-dutch-whispers
pub struct DutchClover;
impl PuzzleSetter for DutchClover {
    type Value = NineStdVal;
    type Overlay = NineStdOverlay;
    type State = NineStd;
    type Ranker = StdRanker;
    type Constraint = MultiConstraint<Self::Value, Self::Overlay, Self::State>;
    

    fn setup() -> (Self::State, Self::Ranker, Self::Constraint) {
        // The given digits in real puzzle but can be overridden in in test.
        Self::setup_with_givens(nine_standard_parse(
            "..5.6.7..\n\
            .........\n\
            ....3.4.5\n\
            .........\n\
            .........\n\
            .........\n\
            2.3.4....\n\
            .........\n\
            ..6.7.8..\n"
        ).unwrap())
    }

    fn setup_with_givens(given: Self::State) -> (Self::State, Self::Ranker, Self::Constraint) {
        let puzzle = given;
        let dw = DutchWhisperBuilder::new(puzzle.get_overlay());
        let whispers = vec![
            dw.polyline(vec![[0, 0], [0, 8], [1, 8], [1, 0], [2, 0], [2, 8]]),
            dw.polyline(vec![[3, 8], [3, 0], [4, 0], [4, 8], [5, 8], [5, 0]]),
            dw.polyline(vec![[6, 8], [6, 0], [7, 0], [7, 8], [8, 8], [8, 0]]),
        ];
        let constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(&puzzle),
            DutchWhisperChecker::new(whispers),
        ]);
        let ranker = StdRanker::new(true, FeatureVec::from_pairs(vec![
            (NUM_POSSIBLE_FEATURE, -100.0),
            (DW_FEATURE, 1.0)
        ]), |_, x, y| x+y);
        (puzzle, ranker, constraint)
    }
}

pub fn main() {
    solve_main::<DutchClover, NineStdTui<DutchClover>>("figures/dutch-clover.png");
}

#[cfg(test)]
mod test {
    use variant_sudoku::{debug::NullObserver, tui::test_util::solve_with_given};
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
        solve_with_given::<DutchClover, _>(sudoku, obs);
    }
}