use variant_sudoku::ranker::{FeatureVec, StdRanker};
use variant_sudoku::constraint::MultiConstraint;
use variant_sudoku::solver::PuzzleSetter;
use variant_sudoku::sudoku::{nine_standard_parse, NineStd, NineStdOverlay, NineStdVal, StdChecker};
use variant_sudoku::thermos::{ThermoBuilder, ThermoChecker, THERMO_BULB_FEATURE, THERMO_FEATURE};
use variant_sudoku::tui::solve_main;
use variant_sudoku::tui_std::NineStdTui;

// https://www.gmpuzzles.com/blog/2023/03/thermo-sudoku-by-thomas-snyder-7/
pub struct ThermoSudokuTs;
impl PuzzleSetter for ThermoSudokuTs {
    type Value = NineStdVal;
    type Overlay = NineStdOverlay;
    type Ranker = StdRanker;
    type Constraint = MultiConstraint<Self::Value, Self::Overlay>;
    
    fn name() -> Option<String> { Some("thermo-sudoku-ts".into()) }

    fn setup() -> (NineStd, Self::Ranker, Self::Constraint) {
        // The given digits in real puzzle but can be overridden in in test.
        Self::setup_with_givens(nine_standard_parse(
            ".........\n\
             .........\n\
             ........8\n\
             .........\n\
             ....6....\n\
             .........\n\
             8........\n\
             .........\n\
             .........\n"
        ).unwrap())
    }

    fn setup_with_givens(given: NineStd) -> (NineStd, Self::Ranker, Self::Constraint) {
        let puzzle = given;
        let tb = ThermoBuilder::<1, 9>::new();
        let thermos = vec![
            tb.polyline(vec![[0, 6], [1, 6], [1, 8], [0, 8], [0, 7]]),
            // Two with the same bulb
            tb.polyline(vec![[1, 1], [3, 1], [3, 3], [1, 3], [1, 2]]),
            tb.polyline(vec![[1, 1], [5, 1], [5, 3], [4, 3]]),
            // Four with the same bulb
            tb.polyline(vec![[5, 6], [5, 5], [3, 5]]),
            tb.polyline(vec![[5, 6], [5, 5], [7, 5], [7, 6]]),
            tb.polyline(vec![[5, 6], [5, 7], [3, 7], [3, 6]]),
            tb.polyline(vec![[5, 6], [5, 7], [7, 7]]),
            tb.polyline(vec![[8, 2], [7, 2], [7, 0], [8, 0], [8, 1]]),
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
    solve_main::<ThermoSudokuTs, NineStdTui<ThermoSudokuTs>>();
}

#[cfg(test)]
mod test {
    use variant_sudoku::{debug::NullObserver, sudoku::nine_standard_parse, tui::test_util::solve_with_given};
    use super::*;

    #[test]
    fn test_thermo_sudoku_ts() {
        let input: &str = "486351297\n\
                           319728456\n\
                           725694318\n\
                           234517869\n\
                           158963742\n\
                           967842135\n\
                           892135674\n\
                           543276981\n\
                           67148952.\n";
        let sudoku = nine_standard_parse(input).unwrap();
        let obs = NullObserver;
        solve_with_given::<ThermoSudokuTs, _>(sudoku, obs);
    }
}