use variant_sudoku::ranker::{FeatureVec, StdRanker};
use variant_sudoku::constraint::MultiConstraint;
use variant_sudoku::solver::PuzzleSetter;
use variant_sudoku::sudoku::{nine_standard_overlay, NineStd, NineStdOverlay, NineStdVal, StdChecker};
use variant_sudoku::cages::{CageBuilder, CageChecker, CAGE_FEATURE};
use variant_sudoku::tui::solve_main;
use variant_sudoku::tui_std::NineStdTui;
use variant_sudoku::xsums::{XSum, XSumChecker, XSumDirection, XSUM_HEAD_FEATURE, XSUM_LN_POSSIBILITIES_FEATURE, XSUM_TAIL_FEATURE};

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000N7H
pub struct DoubleDown;
impl PuzzleSetter for DoubleDown {
    type Value = NineStdVal;
    type Overlay = NineStdOverlay;
    type Ranker = StdRanker<Self::Overlay>;
    type Constraint = MultiConstraint<Self::Value, Self::Overlay>;

    fn name() -> Option<String> { Some("double-down".into()) }

    fn setup() -> (NineStd, Self::Ranker, Self::Constraint) {
        // Real puzzle has no givens
        Self::setup_with_givens(NineStd::new(nine_standard_overlay()))
    }

    fn setup_with_givens(given: NineStd) -> (NineStd, Self::Ranker, Self::Constraint) {
        let puzzle = given;
        let cb = CageBuilder::new(false, puzzle.overlay());
        let cages = vec![
            cb.across(14, [2, 2], 2),
            cb.down(15, [2, 7], 3),
            cb.down(16, [2, 8], 3),
            cb.down(15, [3, 2], 3),
            cb.down(12, [3, 3], 2),
            cb.across(16, [5, 3], 3),
            cb.across(9, [7, 4], 2),
            cb.across(18, [8, 2], 4),
        ];
        let xsums = vec![
            XSum { direction: XSumDirection::RR, index: 2, target: 14 },
            XSum { direction: XSumDirection::RR, index: 5, target: 16 },
            XSum { direction: XSumDirection::RR, index: 8, target: 18 },
            XSum { direction: XSumDirection::CD, index: 2, target: 15 },
            XSum { direction: XSumDirection::CD, index: 5, target: 20 },
            XSum { direction: XSumDirection::CD, index: 8, target: 16 },
            XSum { direction: XSumDirection::RL, index: 3, target: 19 },
            XSum { direction: XSumDirection::RL, index: 7, target: 9 },
            XSum { direction: XSumDirection::CU, index: 3, target: 12 },
            XSum { direction: XSumDirection::CU, index: 7, target: 15 },
        ];
        let constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(&puzzle),
            CageChecker::new(cages),
            XSumChecker::new(xsums),
        ]);
        let ranker = StdRanker::with_additional_weights(FeatureVec::from_pairs(vec![
            (XSUM_TAIL_FEATURE, 100.0),
            (XSUM_HEAD_FEATURE, 5.0),
            (XSUM_LN_POSSIBILITIES_FEATURE, -1.0),
            (CAGE_FEATURE, 1.0)
        ]));
        (puzzle, ranker, constraint)
    }
}

pub fn main() {
    solve_main::<DoubleDown, NineStdTui<_>>();
}

#[cfg(test)]
mod test {
    use variant_sudoku::{debug::NullObserver, sudoku::nine_standard_parse, tui::test_util::solve_with_given};
    use super::*;

    #[test]
    fn test_double_down_solution() {
        let input: &str = "1 7 4|6 2 5|8 9 3\n\
                           9 8 2|7 1 3|5 4 6\n\
                           3 5 6|8 9 4|2 1 7\n\
                           -----+-----+-----\n\
                           8 1 3|9 5 2|7 6 4\n\
                           2 9 7|3 4 6|1 8 5\n\
                           4 6 5|1 7 8|3 2 9\n\
                           -----+-----+-----\n\
                           7 4 8|2 3 9|6 5 1\n\
                           6 3 9|5 8 1|4 7 2\n\
                           5 2 1|4 6 7|9 3 .\n";
        let sudoku = nine_standard_parse(input).unwrap();
        let obs = NullObserver;
        solve_with_given::<DoubleDown, _>(sudoku, obs);
    }
}