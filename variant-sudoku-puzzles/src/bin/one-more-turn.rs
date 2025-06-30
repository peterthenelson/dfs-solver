use variant_sudoku::core::{Error, Overlay, State};
use variant_sudoku::irregular::{IrregularChecker, IrregularOverlay};
use variant_sudoku::ranker::{FeatureVec, StdRanker};
use variant_sudoku::constraint::MultiConstraint;
use variant_sudoku::solver::PuzzleSetter;
use variant_sudoku::sudoku::NineStdVal;
use variant_sudoku::cages::{CageBuilder, CageChecker, CAGE_FEATURE};
use variant_sudoku::tui::solve_main;
use variant_sudoku::tui_std::{DefaultTui, NullOverlayStandardizer};
use variant_sudoku::tui_util::NullConstraintSplitter;

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?chlang=en&id=0007LE
pub struct OneMoreTurn;
impl OneMoreTurn {
    fn make_overlay() -> IrregularOverlay<9, 9> {
        IrregularOverlay::<9, 9>::from_grid(
            "1 1 1|2 2 2|3 3 3\n\
             1 4 1|1 1 2|2 2 3\n\
             1 4 1|4 5 2|3 3 3\n\
             4 4 4|4 5 2|2 3 6\n\
             4 7 5|5 5 5|5 3 6\n\
             4 7 8|8 5 6|6 6 6\n\
             7 7 7|8 5 6|9 6 9\n\
             7 8 8|8 9 9|9 6 9\n\
             7 7 7|8 8 8|9 9 9\n"
        ).unwrap()
    }

    fn parse_state(s: &str) -> Result<State<NineStdVal, IrregularOverlay<9, 9>>, Error> {
        Self::make_overlay().parse_state::<NineStdVal>(s)
    }
}
impl PuzzleSetter for OneMoreTurn {
    type Value = NineStdVal;
    type Overlay = IrregularOverlay<9, 9>;
    type Ranker = StdRanker<Self::Overlay>;
    type Constraint = MultiConstraint<Self::Value, Self::Overlay>;

    fn name() -> Option<String> { Some("one-more-turn".into()) }

    fn setup() -> (State<NineStdVal, Self::Overlay>, Self::Ranker, Self::Constraint) {
        // The given digits in real puzzle but can be overridden in in test.
        Self::setup_with_givens(Self::parse_state(
            ". . 9|. . .|6 . .\n\
             . 1 .|. . .|. 2 .\n\
             8 . .|. . .|. . 5\n\
             -----+-----+-----\n\
             . . .|. . .|. . .\n\
             . . .|. . .|. . .\n\
             . . .|. . .|. . .\n\
             -----+-----+-----\n\
             5 . .|. . .|. . 9\n\
             . 4 .|. . .|. 3 .\n\
             . . 6|. . .|7 . .\n"
        ).unwrap())
    }

    fn setup_with_givens(given: State<NineStdVal, Self::Overlay>) -> (State<NineStdVal, Self::Overlay>, Self::Ranker, Self::Constraint) {
        let puzzle = given;
        let cb = CageBuilder::new(false, puzzle.overlay());
        let cages = vec![cb.across(9, [6, 6], 2)];
        let constraint = MultiConstraint::new(vec_box::vec_box![
            IrregularChecker::new(&puzzle),
            CageChecker::new(cages),
        ]);
        let ranker = StdRanker::with_additional_weights(FeatureVec::from_pairs(vec![
            (CAGE_FEATURE, 1.0),
        ]));
        (puzzle, ranker, constraint)
    }
}

pub fn main() {
    solve_main::<
        OneMoreTurn,
        DefaultTui<
            OneMoreTurn, 9, 9,
            NullOverlayStandardizer<9, 9>,
            NullConstraintSplitter
        >
    >();
}

#[cfg(test)]
mod test {
    use variant_sudoku::{debug::NullObserver, tui::test_util::solve_with_given};
    use super::*;

    #[test]
    fn test_oner_more_turn_solution() {
        let input: &str = "2 7 9|4 8 5|6 1 3\n\
                           6 2 7|5 4 7|9 2 8\n\
                           4 3 8|7 9 3|2 4 5\n\
                           -----+-----+-----\n\
                           4 5 2|8 3 6|1 9 7\n\
                           9 5 3|2 6 4|8 7 1\n\
                           3 8 7|6 1 9|4 5 2\n\
                           -----+-----+-----\n\
                           5 2 4|1 7 8|3 6 9\n\
                           7 4 8|9 2 1|5 3 6\n\
                           1 9 6|3 5 2|7 8 .\n";
        let sudoku = OneMoreTurn::parse_state(input).unwrap();
        let obs = NullObserver;
        solve_with_given::<OneMoreTurn, _>(sudoku, obs);
    }
}