use variant_sudoku::core::{Error, FeatureVec, Overlay, State};
use variant_sudoku::irregular::{IrregularChecker, IrregularOverlay};
use variant_sudoku::ranker::StdRanker;
use variant_sudoku::constraint::MultiConstraint;
use variant_sudoku::solver::PuzzleSetter;
use variant_sudoku::sudoku::NineStdVal;
use variant_sudoku::cages::{CageBuilder, CageChecker, CAGE_FEATURE};
use variant_sudoku::tui::solve_main;
use variant_sudoku::tui_std::{DefaultTui, NullOverlayStandardizer};

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?chlang=en&id=0007LE
pub struct OneMoreTurn;
impl OneMoreTurn {
    fn make_overlay() -> IrregularOverlay<9, 9> {
        IrregularOverlay::<9, 9>::from_grid(
            "111222333\n\
             141112223\n\
             141452333\n\
             444452236\n\
             475555536\n\
             478856666\n\
             777856969\n\
             788899969\n\
             777888999\n"
        ).unwrap()
    }

    fn parse_state(s: &str) -> Result<State<NineStdVal, IrregularOverlay<9, 9>>, Error> {
        Self::make_overlay().parse_state::<NineStdVal>(s)
    }
}
impl PuzzleSetter for OneMoreTurn {
    type Value = NineStdVal;
    type Overlay = IrregularOverlay<9, 9>;
    type Ranker = StdRanker;
    type Constraint = MultiConstraint<Self::Value, Self::Overlay>;

    fn name() -> Option<String> { Some("one-more-turn".into()) }

    fn setup() -> (State<NineStdVal, Self::Overlay>, Self::Ranker, Self::Constraint) {
        // The given digits in real puzzle but can be overridden in in test.
        Self::setup_with_givens(Self::parse_state(
            "..9...6..\n\
             .1.....2.\n\
             8.......5\n\
             .........\n\
             .........\n\
             .........\n\
             5.......9\n\
             .4.....3.\n\
             ..6...7..\n"
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
    solve_main::<OneMoreTurn, DefaultTui<OneMoreTurn, 9, 9, NullOverlayStandardizer<9, 9>>>();
}

#[cfg(test)]
mod test {
    use variant_sudoku::{debug::NullObserver, tui::test_util::solve_with_given};
    use super::*;

    #[test]
    fn test_oner_more_turn_solution() {
        let input: &str = "279485613\n\
                           627547928\n\
                           438793245\n\
                           452836197\n\
                           953264871\n\
                           387619452\n\
                           524178369\n\
                           748921536\n\
                           19635278.\n";
        let sudoku = OneMoreTurn::parse_state(input).unwrap();
        let obs = NullObserver;
        solve_with_given::<OneMoreTurn, _>(sudoku, obs);
    }
}