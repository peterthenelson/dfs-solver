use std::time::Duration;

use variant_sudoku_dfs::core::FeatureVec;
use variant_sudoku_dfs::ranker::{LinearRanker, NUM_POSSIBLE_FEATURE};
use variant_sudoku_dfs::constraint::MultiConstraint;
use variant_sudoku_dfs::solver::FindFirstSolution;
use variant_sudoku_dfs::debug::{DbgObserver, Sample};
use variant_sudoku_dfs::sudoku::{nine_standard_overlay, SState, StandardSudokuChecker};
use variant_sudoku_dfs::cages::{CageBuilder, CageChecker, CAGE_FEATURE};
use variant_sudoku_dfs::xsums::{XSum, XSumDirection, XSumChecker, XSUM_HEAD_FEATURE, XSUM_TAIL_FEATURE};

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000N7H
fn solve(given: Option<SState<9, 9, 1, 9>>, sample_print: Sample) {
    // No given digits in real puzzle but can be passed in in test.
    let mut puzzle = given.unwrap_or(SState::<9, 9, 1, 9>::new());
    let overlay = nine_standard_overlay();
    let cb = CageBuilder::new(false, &overlay);
    let cages = vec![
        cb.cage(14, vec![[2, 2], [2, 3]]),
        cb.cage(15, vec![[2, 7], [3, 7], [4, 7]]),
        cb.cage(16, vec![[2, 8], [3, 8], [4, 8]]),
        cb.cage(15, vec![[3, 2], [4, 2], [5, 2]]),
        cb.cage(12, vec![[3, 3], [4, 3]]),
        cb.cage(16, vec![[5, 3], [5, 4], [5, 5]]),
        cb.cage(9, vec![[7, 4], [7, 5]]),
        cb.cage(18, vec![[8, 2], [8, 3], [8, 4], [8, 5]]),
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
        StandardSudokuChecker::new(&overlay),
        CageChecker::new(cages),
        XSumChecker::new(xsums),
    ]);
    let ranker = LinearRanker::new(FeatureVec::from_pairs(vec![
        (NUM_POSSIBLE_FEATURE, -100.0),
        (XSUM_TAIL_FEATURE, 10.0),
        (XSUM_HEAD_FEATURE, 5.0),
        (CAGE_FEATURE, 1.0)
    ]));
    let mut dbg = DbgObserver::new();
    dbg.sample_print(sample_print)
        .sample_stats("figures/double-down-stats.png", Sample::time(Duration::from_secs(30)));
    let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, Some(&mut dbg));
    let maybe_solution = finder.solve().expect("Puzzle solver returned an error:");
    println!("Solution:\n{}", maybe_solution.expect("No solution found!").get_state().serialize());
}

pub fn main() {
    solve(None, Sample::every_n(100000));
}

#[cfg(test)]
mod test {
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
        let sudoku: SState<9,9, 1,9> = SState::parse(input).unwrap();
        solve(Some(sudoku), Sample::never());
    }
}