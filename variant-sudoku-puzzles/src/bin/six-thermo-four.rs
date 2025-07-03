use variant_sudoku::ranker::{FeatureVec, StdRanker};
use variant_sudoku::constraint::MultiConstraint;
use variant_sudoku::solver::PuzzleSetter;
use variant_sudoku::sudoku::{six_standard_parse, SixStd, SixStdOverlay, SixStdVal, StdChecker};
use variant_sudoku::thermos::{ThermoBuilder, ThermoChecker, THERMO_BULB_FEATURE, THERMO_FEATURE};
use variant_sudoku::tui::solve_main;
use variant_sudoku::tui_std::SixStdTui;

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000O0Q
pub struct SixThermoFour;
impl PuzzleSetter for SixThermoFour {
    type Value = SixStdVal;
    type Overlay = SixStdOverlay;
    type Ranker = StdRanker<Self::Overlay>;
    type Constraint = MultiConstraint<Self::Value, Self::Overlay>;
    
    fn name() -> Option<String> { Some("six-thermo-four".into()) }

    fn setup() -> (SixStd, Self::Ranker, Self::Constraint) {
        // The given digits in real puzzle but can be overridden in in test.
        Self::setup_with_givens(six_standard_parse(
            ". . .|. . .\n\
             . . .|. . .\n\
             -----+-----\n\
             . . .|2 . .\n\
             . . .|. . .\n\
             -----+-----\n\
             . . .|. . .\n\
             . . .|. . .\n"
        ).unwrap())
    }

    fn setup_with_givens(given: SixStd) -> (SixStd, Self::Ranker, Self::Constraint) {
        let puzzle = given;
        let tb = ThermoBuilder::<1, 9>::new();
        let thermos = vec![
            tb.polyline(vec![[0, 3], [1, 3], [1, 4], [2, 4]]),
            tb.polyline(vec![[1, 2], [1, 1], [2, 1], [2, 0]]),
            tb.polyline(vec![[4, 3], [4, 4], [3, 4], [3, 5]]),
            tb.polyline(vec![[5, 2], [4, 2], [4, 1], [3, 1]]),
        ];
        let constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(&puzzle),
            ThermoChecker::new(thermos),
        ]);
        let ranker = StdRanker::with_additional_weights(FeatureVec::from_pairs(vec![
            (THERMO_BULB_FEATURE, 2.0),
            (THERMO_FEATURE, 1.0),
        ]));
        (puzzle, ranker, constraint)
    }
}

pub fn main() {
    solve_main::<SixThermoFour, SixStdTui<SixThermoFour>>();
}

#[cfg(test)]
mod test {
    use variant_sudoku::{debug::NullObserver, sudoku::six_standard_parse, tui::test_util::solve_with_given};
    use super::*;

    #[test]
    fn test_six_thermo_four() {
        let input: &str = "5 4 6|3 1 2\n\
                           3 2 1|4 5 6\n\
                           -----+-----\n\
                           4 3 5|2 6 1\n\
                           1 6 2|5 3 4\n\
                           -----+-----\n\
                           6 5 4|1 2 3\n\
                           2 1 3|6 4 .\n";
        let sudoku = six_standard_parse(input).unwrap();
        let obs = NullObserver;
        solve_with_given::<SixThermoFour, _>(sudoku, obs);
    }
}