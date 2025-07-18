use variant_sudoku::kropki::{KropkiBuilder, KropkiChecker, KROPKI_BLACK_FEATURE};
use variant_sudoku::ranker::{FeatureVec, StdRanker};
use variant_sudoku::constraint::MultiConstraint;
use variant_sudoku::solver::PuzzleSetter;
use variant_sudoku::sudoku::{nine_standard_overlay, NineStd, NineStdOverlay, NineStdVal, StdChecker};
use variant_sudoku::cages::{CageBuilder, CageChecker, CAGE_FEATURE};
use variant_sudoku::tui::solve_main;
use variant_sudoku::tui_std::NineStdTui;

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000NRV
pub struct ThreeColorTheorem;
impl PuzzleSetter for ThreeColorTheorem {
    type Value = NineStdVal;
    type Overlay = NineStdOverlay;
    type Ranker = StdRanker<Self::Overlay>;
    type Constraint = MultiConstraint<Self::Value, Self::Overlay>;

    fn name() -> Option<String> { Some("three-color-theorem".into()) }

    fn setup() -> (NineStd, Self::Ranker, Self::Constraint) {
        // No given digits in real puzzle but can be passed in in test.
        Self::setup_with_givens(NineStd::new(nine_standard_overlay()))
    }

    fn setup_with_givens(given: NineStd) -> (NineStd, Self::Ranker, Self::Constraint) {
        let puzzle = given;
        let cb = CageBuilder::new(true, puzzle.overlay());
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
        let kb = KropkiBuilder::new(puzzle.overlay());
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
            StdChecker::new(puzzle.overlay()),
            CageChecker::new(cages),
            KropkiChecker::new(kropkis),
        ]);
        let ranker = StdRanker::with_additional_weights(FeatureVec::from_pairs(vec![
            (CAGE_FEATURE, 1.0),
            (KROPKI_BLACK_FEATURE, 1.0),
        ]));
        (puzzle, ranker, constraint)
    }
}

pub fn main() {
    solve_main::<ThreeColorTheorem, NineStdTui<ThreeColorTheorem>>();
}

#[cfg(test)]
mod test {
    use variant_sudoku::{debug::NullObserver, sudoku::nine_standard_parse, tui::test_util::solve_with_given};
    use super::*;

    #[test]
    fn test_three_color_theorem_solution() {
        let input: &str = "4 5 6|3 9 2|1 7 8\n\
                           2 7 3|8 5 1|6 9 4\n\
                           1 9 8|4 7 6|3 5 2\n\
                           -----+-----+-----\n\
                           7 8 4|5 6 3|9 2 1\n\
                           6 1 5|2 4 9|8 3 7\n\
                           3 2 9|1 8 7|4 6 5\n\
                           -----+-----+-----\n\
                           5 6 2|9 1 4|7 8 3\n\
                           9 3 1|7 2 8|5 4 6\n\
                           8 4 7|6 3 5|2 1 .\n";
        let sudoku = nine_standard_parse(input).unwrap();
        let obs = NullObserver;
        solve_with_given::<ThreeColorTheorem, _>(sudoku, obs);
    }
}