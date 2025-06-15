use std::time::Duration;
use variant_sudoku_dfs::core::FeatureVec;
use variant_sudoku_dfs::ranker::{OverlaySensitiveLinearRanker, NUM_POSSIBLE_FEATURE};
use variant_sudoku_dfs::constraint::MultiConstraint;
use variant_sudoku_dfs::solver::PuzzleSetter;
use variant_sudoku_dfs::debug::{DbgObserver, Sample};
use variant_sudoku_dfs::sudoku::{nine_standard_overlay, NineStd, StandardSudokuChecker};
use variant_sudoku_dfs::cages::{CageBuilder, CageChecker, CAGE_FEATURE};
use variant_sudoku_dfs::tui::{solve_cli};
use variant_sudoku_dfs::xsums::{XSum, XSumDirection, XSumChecker, XSUM_HEAD_FEATURE, XSUM_TAIL_FEATURE};

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000N7H
pub struct DoubleDown;
impl PuzzleSetter for DoubleDown {
    type U = u8;
    type State = NineStd;
    type Ranker = OverlaySensitiveLinearRanker;
    type Constraint = MultiConstraint<u8, NineStd>;

    fn setup() -> (NineStd, OverlaySensitiveLinearRanker, MultiConstraint<u8, NineStd>) {
        // Real puzzle has no givens
        Self::setup_with_givens(NineStd::new(nine_standard_overlay()))
    }

    fn setup_with_givens(given: NineStd) -> (NineStd, OverlaySensitiveLinearRanker, MultiConstraint<u8, NineStd>) {
        let puzzle = given;
        let cb = CageBuilder::new(false, puzzle.get_overlay());
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
            StandardSudokuChecker::new(&puzzle),
            CageChecker::new(cages),
            XSumChecker::new(xsums),
        ]);
        let ranker = OverlaySensitiveLinearRanker::new(FeatureVec::from_pairs(vec![
            (NUM_POSSIBLE_FEATURE, -100.0),
            (XSUM_TAIL_FEATURE, 10.0),
            (XSUM_HEAD_FEATURE, 5.0),
            (CAGE_FEATURE, 1.0)
        ]), |_, x, y| x+y);
        (puzzle, ranker, constraint)
    }
}

pub fn main() {
    let mut dbg = DbgObserver::new();
    dbg.sample_print(Sample::every_n(100000))
        .sample_stats("figures/double-down-stats.png", Sample::time(Duration::from_secs(30)));
    solve_cli::<DoubleDown, _>(dbg);
}

#[cfg(all(test, feature = "test-util"))]
mod test {
    use variant_sudoku_dfs::{debug::NullObserver, sudoku::nine_standard_parse, tui::test_util::solve_with_given};
    use super::*;

    #[test]
    fn test_double_down_solution() {
        let input: &str = "174625893\n\
                           982713546\n\
                           356894217\n\
                           813952764\n\
                           297346185\n\
                           465178329\n\
                           748239651\n\
                           639581472\n\
                           52146793.\n";
        let sudoku = nine_standard_parse(input).unwrap();
        let obs = NullObserver;
        solve_with_given::<DoubleDown, _>(sudoku, obs);
    }
}