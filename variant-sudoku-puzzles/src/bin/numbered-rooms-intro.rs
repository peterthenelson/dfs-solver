use variant_sudoku::ranker::{FeatureVec, StdRanker};
use variant_sudoku::constraint::MultiConstraint;
use variant_sudoku::kropki::{KropkiBuilder, KropkiChecker, KROPKI_BLACK_FEATURE, KROPKI_WHITE_FEATURE};
use variant_sudoku::numbered_rooms::{NumberedRoom, NumberedRoomChecker, NumberedRoomDirection, NUMBERED_ROOM_FEATURE};
use variant_sudoku::solver::PuzzleSetter;
use variant_sudoku::sudoku::{nine_standard_overlay, NineStd, NineStdOverlay, NineStdVal, StdChecker};
use variant_sudoku::tui::solve_main;
use variant_sudoku::tui_std::NineStdTui;

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000O1U
pub struct NumberedRoomsIntro;
impl PuzzleSetter for NumberedRoomsIntro {
    type Value = NineStdVal;
    type Overlay = NineStdOverlay;
    type Ranker = StdRanker<Self::Overlay>;
    type Constraint = MultiConstraint<Self::Value, Self::Overlay>;

    fn name() -> Option<String> { Some("numbered-rooms-intro".into()) }

    fn setup() -> (NineStd, Self::Ranker, Self::Constraint) {
        // Real puzzle has no givens
        Self::setup_with_givens(NineStd::new(nine_standard_overlay()))
    }

    fn setup_with_givens(given: NineStd) -> (NineStd, Self::Ranker, Self::Constraint) {
        let puzzle = given;
        let kropkis = {
            let kb = KropkiBuilder::new(puzzle.overlay());
            vec![
                kb.w_across([0, 0]),
                kb.w_across([0, 7]),
                kb.b_across([3, 3]),
                kb.w_down([4, 3]),
                kb.b_across([5, 1]),
                kb.b_down([6, 4]),
            ]
        };
        let rooms = vec![
            // Across
            NumberedRoom::new(2, 1, NumberedRoomDirection::RB),
            NumberedRoom::new(2, 2, NumberedRoomDirection::RR),
            NumberedRoom::new(9, 2, NumberedRoomDirection::RL),
            NumberedRoom::new(1, 3, NumberedRoomDirection::RL),
            NumberedRoom::new(3, 5, NumberedRoomDirection::RB),
            NumberedRoom::new(4, 7, NumberedRoomDirection::RL),
            NumberedRoom::new(5, 8, NumberedRoomDirection::RR),
            // Up/Down
            NumberedRoom::new(3, 0, NumberedRoomDirection::CD),
            NumberedRoom::new(3, 1, NumberedRoomDirection::CD),
            NumberedRoom::new(1, 1, NumberedRoomDirection::CU),
            NumberedRoom::new(3, 2, NumberedRoomDirection::CB),
            NumberedRoom::new(1, 3, NumberedRoomDirection::CB),
            NumberedRoom::new(9, 4, NumberedRoomDirection::CD),
            NumberedRoom::new(2, 6, NumberedRoomDirection::CD),
            NumberedRoom::new(3, 6, NumberedRoomDirection::CU),
            NumberedRoom::new(2, 7, NumberedRoomDirection::CD),
            NumberedRoom::new(2, 8, NumberedRoomDirection::CD),
        ];
        let constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(puzzle.overlay()),
            KropkiChecker::new(kropkis),
            NumberedRoomChecker::new(rooms),
        ]);
        let ranker = StdRanker::with_additional_weights(FeatureVec::from_pairs(vec![
            (NUMBERED_ROOM_FEATURE, 5.0),
            (KROPKI_BLACK_FEATURE, 1.0),
            (KROPKI_WHITE_FEATURE, 1.0),
        ]));
        (puzzle, ranker, constraint)
    }
}

pub fn main() {
    solve_main::<NumberedRoomsIntro, NineStdTui<_>>();
}

#[cfg(test)]
mod test {
    use variant_sudoku::{debug::NullObserver, sudoku::nine_standard_parse, tui::test_util::solve_with_given};
    use super::*;

    #[test]
    fn test_numbered_rooms_intro_solution() {
        let input: &str = "7 6 2|1 5 8|9 3 4\n\
                           4 5 3|2 7 9|1 8 6\n\
                           8 1 9|3 6 4|5 2 7\n\
                           -----+-----+-----\n\
                           5 9 7|4 8 3|6 1 2\n\
                           1 8 4|6 9 2|7 5 3\n\
                           2 3 6|5 1 7|4 9 8\n\
                           -----+-----+-----\n\
                           3 4 5|7 2 1|8 6 9\n\
                           9 2 1|8 4 6|3 7 5\n\
                           6 7 8|9 3 5|2 4 .\n";
        let sudoku = nine_standard_parse(input).unwrap();
        let obs = NullObserver;
        solve_with_given::<NumberedRoomsIntro, _>(sudoku, obs);
    }
}