use variant_sudoku::ranker::{FeatureVec, StdRanker};
use variant_sudoku::constraint::MultiConstraint;
use variant_sudoku::solver::PuzzleSetter;
use variant_sudoku::sudoku::{nine_standard_overlay, NineStd, NineStdOverlay, NineStdVal, StdChecker};
use variant_sudoku::thermos::{ThermoBuilder, ThermoChecker, THERMO_BULB_FEATURE, THERMO_FEATURE};
use variant_sudoku::tui::solve_main;
use variant_sudoku::tui_std::NineStdTui;

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000NU8
pub struct TopRightBoiThermo;
impl PuzzleSetter for TopRightBoiThermo {
    type Value = NineStdVal;
    type Overlay = NineStdOverlay;
    type Ranker = StdRanker<Self::Overlay>;
    type Constraint = MultiConstraint<Self::Value, Self::Overlay>;
    
    fn name() -> Option<String> { Some("top-right-boi-thermo".into()) }

    fn setup() -> (NineStd, Self::Ranker, Self::Constraint) {
        // No givens
        Self::setup_with_givens(NineStd::new(nine_standard_overlay()))
    }

    fn setup_with_givens(given: NineStd) -> (NineStd, Self::Ranker, Self::Constraint) {
        let puzzle = given;
        let tb = ThermoBuilder::<1, 9>::new();
        let thermos = vec![
            tb.polyline(vec![[0, 5], [3, 2]]),
            tb.polyline(vec![[0, 6], [3, 3]]),
            tb.polyline(vec![[1, 3], [2, 2]]),
            tb.polyline(vec![[1, 6], [6, 1]]),
            tb.polyline(vec![[2, 6], [4, 4]]),
            tb.polyline(vec![[2, 7], [8, 1]]),
            tb.polyline(vec![[2, 8], [5, 5]]),
            tb.polyline(vec![[3, 8], [7, 4]]),
            tb.polyline(vec![[4, 2], [5, 1]]),
            tb.polyline(vec![[5, 8], [7, 6]]),
        ];
        let constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(puzzle.overlay()),
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
    solve_main::<TopRightBoiThermo, NineStdTui<TopRightBoiThermo>>();
}

#[cfg(test)]
mod test {
    use variant_sudoku::{debug::NullObserver, sudoku::nine_standard_parse, tui::test_util::solve_with_given};
    use super::*;

    #[test]
    fn test_top_right_boi_thermo() {
        let input: &str = "2 4 3|9 8 1|5 7 6\n\
                           7 5 1|3 2 6|4 9 8\n\
                           8 6 9|4 7 5|1 2 3\n\
                           -----+-----+-----\n\
                           9 7 5|8 6 2|3 4 1\n\
                           6 1 2|7 3 4|8 5 9\n\
                           4 3 8|1 5 9|7 6 2\n\
                           -----+-----+-----\n\
                           5 9 4|6 1 8|2 3 7\n\
                           1 2 7|5 9 3|6 8 4\n\
                           3 8 6|2 4 7|9 1 .\n";
        let sudoku = nine_standard_parse(input).unwrap();
        let obs = NullObserver;
        solve_with_given::<TopRightBoiThermo, _>(sudoku, obs);
    }
}