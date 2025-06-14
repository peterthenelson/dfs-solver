use std::time::Duration;
use variant_sudoku_dfs::core::FeatureVec;
use variant_sudoku_dfs::ranker::{OverlaySensitiveLinearRanker, NUM_POSSIBLE_FEATURE};
use variant_sudoku_dfs::constraint::MultiConstraint;
use variant_sudoku_dfs::solver::{FindFirstSolution, StepObserver};
use variant_sudoku_dfs::debug::{DbgObserver, Sample};
use variant_sudoku_dfs::sudoku::{nine_standard_overlay, SState, StandardSudokuChecker, StandardSudokuOverlay};
use variant_sudoku_dfs::cages::{CageBuilder, CageChecker, CAGE_FEATURE};
use variant_sudoku_dfs::xsums::{XSum, XSumDirection, XSumChecker, XSUM_HEAD_FEATURE, XSUM_TAIL_FEATURE};

type NineStd = SState<9, 9, 1, 9, StandardSudokuOverlay<9, 9>>;

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000N7H
fn solve<D: StepObserver<u8, NineStd>>(
    given: Option<NineStd>,
    mut observer: D,
) {
    // No given digits in real puzzle but can be passed in in test.
    let mut puzzle = given.unwrap_or(
        SState::<9, 9, 1, 9, _>::new(nine_standard_overlay())
    );
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
    let mut constraint = MultiConstraint::new(vec_box::vec_box![
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
    let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, Some(&mut observer));
    let maybe_solution = finder.solve().expect("Puzzle solver returned an error:");
    println!("Solution:\n{}", maybe_solution.expect("No solution found!").get_state().serialize());
}

pub fn main() {
    let mut dbg = DbgObserver::new();
    dbg.sample_print(Sample::every_n(100000))
        .sample_stats("figures/double-down-stats.png", Sample::time(Duration::from_secs(30)));
    solve(None, dbg);
}

#[cfg(test)]
mod test {
    use variant_sudoku_dfs::{debug::NullObserver, sudoku::nine_standard_parse};
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
        solve(Some(sudoku), obs);
    }
}