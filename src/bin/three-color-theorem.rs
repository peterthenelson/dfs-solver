use std::time::Duration;
use variant_sudoku_dfs::core::FeatureVec;
use variant_sudoku_dfs::kropki::{KropkiBuilder, KropkiChecker, KROPKI_BLACK_FEATURE};
use variant_sudoku_dfs::ranker::{OverlaySensitiveLinearRanker, NUM_POSSIBLE_FEATURE};
use variant_sudoku_dfs::constraint::MultiConstraint;
use variant_sudoku_dfs::solver::PuzzleSetter;
use variant_sudoku_dfs::debug::{DbgObserver, Sample};
use variant_sudoku_dfs::sudoku::{nine_standard_overlay, NineStd, StandardSudokuChecker};
use variant_sudoku_dfs::cages::{CageBuilder, CageChecker, CAGE_FEATURE};
use variant_sudoku_dfs::tui::solve_cli;

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000NRV
pub struct ThreeColorTheorem;
impl PuzzleSetter for ThreeColorTheorem {
    type U = u8;
    type State = NineStd;
    type Ranker = OverlaySensitiveLinearRanker;
    type Constraint = MultiConstraint<u8, NineStd>;

    fn setup() -> (NineStd, OverlaySensitiveLinearRanker, MultiConstraint<u8, NineStd>) {
        // No given digits in real puzzle but can be passed in in test.
        Self::setup_with_givens(NineStd::new(nine_standard_overlay()))
    }

    fn setup_with_givens(given: NineStd) -> (NineStd, OverlaySensitiveLinearRanker, MultiConstraint<u8, NineStd>) {
        let puzzle = given;
        let cb = CageBuilder::new(true, puzzle.get_overlay());
        let cages = vec![
            cb.across(15, [0, 0], 3),
            cb.nosum(vec![[0, 3], [0, 4], [0, 5]]),
            cb.nosum(vec![[0, 6], [0, 7], [0, 8]]),
            cb.nosum(vec![[1, 0], [1, 1], [1, 2]]),
            cb.nosum(vec![[1, 3], [1, 4], [1, 5]]),
            cb.across(19, [1, 6], 3),
            cb.across(18, [2, 0], 3),
            cb.across(17, [2, 3], 3),
            cb.nosum(vec![[2, 6], [2, 7], [2, 8]]),
            cb.nosum(vec![[4, 0], [3, 0], [3, 1]]),
            cb.sum(10, vec![[4, 1], [4, 2], [3, 2]]),
            cb.sum(13, vec![[4, 3], [3, 3], [3, 4]]),
            cb.sum(16, vec![[4, 4], [4, 5], [3, 5]]),
            cb.nosum(vec![[4, 6], [3, 6], [3, 7]]),
            cb.sum(11, vec![[4, 7], [4, 8], [3, 8]]),
            cb.sum(14, vec![[5, 0], [6, 0], [6, 1]]),
            cb.nosum(vec![[5, 1], [5, 2], [5, 3]]),
            cb.nosum(vec![[5, 4], [5, 5], [5, 6]]),
            cb.nosum(vec![[5, 7], [5, 8], [6, 8]]),
            cb.nosum(vec![[6, 2], [6, 3], [6, 4]]),
            cb.nosum(vec![[6, 5], [6, 6], [6, 7]]),
            cb.sum(20, vec![[8, 0], [7, 0], [7, 1]]),
            cb.sum(12, vec![[8, 1], [8, 2], [7, 2]]),
            cb.nosum(vec![[8, 3], [7, 3], [7, 4]]),
            cb.sum(16, vec![[8, 4], [8, 5], [7, 5]]),
            cb.nosum(vec![[8, 6], [7, 6], [7, 7]]),
            cb.sum(16, vec![[8, 7], [8, 8], [7, 8]]),
        ];
        let kb = KropkiBuilder::new(puzzle.get_overlay());
        let kropkis = vec![
            kb.b_chain(vec![[0, 0], [1, 0], [2, 0]]),
            kb.b_chain(vec![[1, 2], [0, 2], [0, 3]]),
            kb.b_chain(vec![[2, 2], [2, 3], [1, 3]]),
            kb.b_chain(vec![[1, 5], [0, 5], [0, 6]]),
            kb.b_chain(vec![[2, 5], [2, 6], [1, 6]]),
            kb.b_down([0, 8]),
            kb.b_across([3, 1]),
            kb.b_across([3, 4]),
            kb.b_chain(vec![[3, 7], [3, 8], [2, 8]]),
            kb.b_down([4, 0]),
            kb.b_down([4, 1]),
            kb.b_down([4, 3]),
            kb.b_down([4, 4]),
            kb.b_down([4, 6]),
            kb.b_down([4, 7]),
            kb.b_down([6, 1]),
            kb.b_down([6, 2]),
            kb.b_down([6, 4]),
            kb.b_down([6, 5]),
            kb.b_down([6, 7]),
            kb.b_down([6, 8]),
            kb.b_across([8, 0]),
            kb.b_across([8, 3]),
            kb.b_across([8, 6]),
        ];
        let constraint = MultiConstraint::new(vec_box::vec_box![
            StandardSudokuChecker::new(&puzzle),
            CageChecker::new(cages),
            KropkiChecker::new(kropkis),
        ]);
        let ranker = OverlaySensitiveLinearRanker::new(FeatureVec::from_pairs(vec![
            (NUM_POSSIBLE_FEATURE, -100.0),
            (CAGE_FEATURE, 1.0),
            (KROPKI_BLACK_FEATURE, 1.0),
        ]), |_, x, y| x+y);
        (puzzle, ranker, constraint)
    }
}

pub fn main() {
    let mut dbg = DbgObserver::new();
    dbg.sample_print(Sample::every_n(10000))
        .sample_stats("figures/three-color-theorem.png", Sample::time(Duration::from_secs(30)));
    solve_cli::<ThreeColorTheorem, _>(dbg);
}

#[cfg(all(test, feature = "test-util"))]
mod test {
    use variant_sudoku_dfs::{debug::NullObserver, sudoku::nine_standard_parse, tui::test_util::solve_with_given};
    use super::*;

    #[test]
    fn test_three_color_theorem_solution() {
        let input: &str = "456392178\n\
                           273851694\n\
                           198476352\n\
                           784563921\n\
                           615249837\n\
                           329187465\n\
                           562914783\n\
                           931728546\n\
                           84763521.\n";
        let sudoku = nine_standard_parse(input).unwrap();
        let obs = NullObserver;
        solve_with_given::<ThreeColorTheorem, _>(sudoku, obs);
    }
}