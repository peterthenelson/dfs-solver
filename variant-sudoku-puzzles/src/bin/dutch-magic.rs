use variant_sudoku::dutch_whispers::{DutchWhisperBuilder, DutchWhisperChecker};
use variant_sudoku::magic_squares::{MagicSquare, MagicSquareChecker, MS_FEATURE};
use variant_sudoku::ranker::{FeatureVec, StdRanker};
use variant_sudoku::constraint::MultiConstraint;
use variant_sudoku::solver::PuzzleSetter;
use variant_sudoku::sudoku::{nine_standard_overlay, NineStd, NineStdOverlay, NineStdVal, StdChecker};
use variant_sudoku::cages::{CageBuilder, CageChecker, CAGE_FEATURE};
use variant_sudoku::tui::solve_main;
use variant_sudoku::tui_std::NineStdTui;

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000NRF
pub struct DutchMagic;
impl PuzzleSetter for DutchMagic {
    type Value = NineStdVal;
    type Overlay = NineStdOverlay;
    type Ranker = StdRanker;
    type Constraint = MultiConstraint<Self::Value, Self::Overlay>;

    fn name() -> Option<String> { Some("dutch-magic".into()) }

    fn setup() -> (NineStd, Self::Ranker, Self::Constraint) {
        // No given digits in real puzzle but can be passed in in test.
        Self::setup_with_givens(NineStd::new(nine_standard_overlay()))
    }

    fn setup_with_givens(given: NineStd) -> (NineStd, Self::Ranker, Self::Constraint) {
        let puzzle = given;
        let cb = CageBuilder::new(false, puzzle.overlay());
        let cages = vec![
            cb.v([2, 4], [3, 4]),
            cb.v([4, 5], [4, 6]),
        ];
        let dw = DutchWhisperBuilder::new(puzzle.overlay());
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
        let constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(&puzzle),
            CageChecker::new(cages),
            DutchWhisperChecker::new(whispers),
            MagicSquareChecker::new(squares),
        ]);
        let ranker = StdRanker::with_additional_weights(FeatureVec::from_pairs(vec![
            (CAGE_FEATURE, 1.0),
            (MS_FEATURE, 1.0),
        ]));
        (puzzle, ranker, constraint)
    }
}

pub fn main() {
    solve_main::<DutchMagic, NineStdTui<DutchMagic>>();
}

#[cfg(test)]
mod test {
    use variant_sudoku::{debug::NullObserver, sudoku::nine_standard_parse, tui::test_util::solve_with_given};
    use super::*;

    #[test]
    fn test_dutch_magic_solution() {
        let input: &str = "1 9 5|6 4 3|8 2 7\n\
                           6 2 7|8 9 5|1 4 3\n\
                           4 3 8|1 2 7|6 9 5\n\
                           -----+-----+-----\n\
                           9 5 1|4 3 8|7 6 2\n\
                           2 7 6|9 5 1|4 3 8\n\
                           8 4 3|2 7 6|9 5 1\n\
                           -----+-----+-----\n\
                           5 1 4|3 8 9|2 7 6\n\
                           7 6 9|5 1 2|3 8 4\n\
                           3 8 2|7 6 4|5 1 .\n";
        let sudoku = nine_standard_parse(input).unwrap();
        let obs = NullObserver;
        solve_with_given::<DutchMagic, _>(sudoku, obs);
    }
}