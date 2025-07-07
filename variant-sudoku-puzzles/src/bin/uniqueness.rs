use variant_sudoku::cages::{CageBuilder, CageChecker, CAGE_FEATURE};
use variant_sudoku::core::{ConstraintResult, Index, Key, State};
use variant_sudoku::kropki::{KropkiBuilder, KropkiChecker, KROPKI_BLACK_FEATURE, KROPKI_WHITE_FEATURE};
use variant_sudoku::parity_shading::{ParityShadingBuilder, ParityShadingChecker};
use variant_sudoku::ranker::{FeatureVec, StdRanker};
use variant_sudoku::constraint::MultiConstraint;
use variant_sudoku::region_constraint::{RegionConstraint, RegionContraintBuilder};
use variant_sudoku::simple_constraint::SimpleConstraint;
use variant_sudoku::solver::PuzzleSetter;
use variant_sudoku::sudoku::{nine_standard_empty, nine_standard_overlay, NineStd, NineStdOverlay, NineStdVal, StdChecker};
use variant_sudoku::tui::solve_main;
use variant_sudoku::tui_std::NineStdTui;

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000NV8
pub struct Uniqueness;
impl PuzzleSetter for Uniqueness {
    type Value = NineStdVal;
    type Overlay = NineStdOverlay;
    type Ranker = StdRanker<Self::Overlay>;
    type Constraint = MultiConstraint<Self::Value, Self::Overlay>;
    
    fn name() -> Option<String> { Some("uniqueness".into()) }

    fn setup() -> (NineStd, Self::Ranker, Self::Constraint) {
        // There are no given digits in real puzzle but this can be overridden
        // in a test.
        Self::setup_with_givens(nine_standard_empty())
    }

    fn setup_with_givens(given: NineStd) -> (NineStd, Self::Ranker, Self::Constraint) {
        let mut overlay = nine_standard_overlay();
        let parity_regions = {
            let mut rb = RegionContraintBuilder::new(&mut overlay, "PARITY");
            vec![
                // Paritition shadings
                rb.region(vec![
                    [0, 0], [0, 5], [1, 6],
                    [4, 1], [5, 7], [6, 0],
                    [7, 1], [8, 2], [8, 6],
                ]),
            ]
        };
        let other_regions = {
            let mut rb = RegionContraintBuilder::new(&mut overlay, "OTHER");
            vec![
                // Kropki white dots
                rb.region(vec![
                    [0, 5], [1, 5], [1, 6], [1, 7],
                    [4, 0], [4, 1],
                    [7, 2], [8, 2], [8, 3],
                ]),
                // Kropki black dots
                rb.region(vec![
                    [2, 0], [2, 1], [2, 2],
                    [3, 6], [3, 7],
                ]),
                // Cages (including the simple constraint and a targetless box)
                rb.region(vec![
                    [1, 0], [2, 4], [2, 5],
                    [3, 4], [3, 5], [4, 4],
                    [4, 5], [5, 7], [5, 8],
                ]),
            ]
        };
        let shadings = {
            let pb = ParityShadingBuilder::new();
            vec![
                pb.even([0, 0]),
                pb.odd([0, 5]),
                pb.odd([1, 6]),
                pb.odd([4, 1]),
                pb.odd([5, 7]),
                pb.odd([6, 0]),
                pb.even([7, 1]),
                pb.even([8, 2]),
                pb.even([8, 6]),
            ]
        };
        let kropkis = {
            let kb = KropkiBuilder::new(&overlay);
            vec![
                kb.w_polyline(vec![[0, 5], [1, 5], [1, 7]]),
                kb.b_polyline(vec![[2, 0], [2, 2]]),
                kb.b_across([3, 6]),
                kb.w_across([4, 0]),
                kb.w_polyline(vec![[7, 2], [8, 2], [8, 3]]),
            ]
        };
        let cages = {
            let cb = CageBuilder::new(true, &overlay);
            vec![cb.rect(37, [2, 4], [4, 5])]
        };
        let constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(&overlay),
            RegionConstraint::new("PARITY", parity_regions),
            RegionConstraint::new("OTHER", other_regions),
            ParityShadingChecker::new(shadings),
            KropkiChecker::new(kropkis),
            CageChecker::new(cages),
            SimpleConstraint {
                name: Some("[5,7] + [5,8] > 6".to_string()),
                check: |puzzle: &State<NineStdVal, NineStdOverlay>, _| {
                    if let Some((a, b)) = puzzle.get([5, 7]).zip(puzzle.get([5, 8])) {
                        if a.val() + b.val() <= 6 {
                            return ConstraintResult::Contradiction(Key::register("CELLS_LTE_6"));
                        }
                    }
                    ConstraintResult::Ok
                },
                debug_str: None,
                debug_at: None,
                debug_highlight: Some(|_, cell: Index| {
                    if cell == [5, 7] || cell == [5, 8] {
                        Some((200, 200, 0))
                    } else {
                        None
                    }
                }),
            },
        ]);
        let puzzle = given.clone_with_overlay(overlay);
        let ranker = StdRanker::with_additional_weights(FeatureVec::from_pairs(vec![
            (CAGE_FEATURE, 1.0),
            (KROPKI_BLACK_FEATURE, 1.0),
            (KROPKI_WHITE_FEATURE, 2.0),
        ]));
        (puzzle, ranker, constraint)
    }
}

pub fn main() {
    solve_main::<Uniqueness, NineStdTui<Uniqueness>>();
}

#[cfg(test)]
mod test {
    use variant_sudoku::{debug::NullObserver, sudoku::nine_standard_parse, tui::test_util::solve_with_given};
    use super::*;

    #[test]
    fn test_uniqueness_solution() {
        let input: &str = "6 3 7|1 5 9|4 2 8\n\
                           1 5 9|4 2 8|7 6 3\n\
                           8 4 2|6 7 3|1 9 5\n\
                           -----+-----+-----\n\
                           5 7 8|2 9 4|6 3 1\n\
                           2 1 3|5 8 6|9 7 4\n\
                           4 9 6|7 3 1|8 5 2\n\
                           -----+-----+-----\n\
                           3 2 1|8 6 7|5 4 9\n\
                           7 8 5|9 4 2|3 1 6\n\
                           9 6 4|3 1 5|2 8 .\n";
        let sudoku = nine_standard_parse(input).unwrap();
        let obs = NullObserver;
        solve_with_given::<Uniqueness, _>(sudoku, obs);
    }
}