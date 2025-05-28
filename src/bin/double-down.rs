use variant_sudoku_dfs::core::FeatureVec;
use variant_sudoku_dfs::ranker::{LinearRanker, NUM_POSSIBLE_FEATURE};
use variant_sudoku_dfs::constraint::ConstraintConjunction;
use variant_sudoku_dfs::solver::{FindFirstSolution, SamplingDbgObserver};
use variant_sudoku_dfs::sudoku::{nine_standard_checker, SState};
use variant_sudoku_dfs::cages::{CageBuilder, CageChecker, CAGE_FEATURE};
use variant_sudoku_dfs::xsums::{XSum, XSumDirection, XSumChecker, XSUM_HEAD_FEATURE, XSUM_TAIL_FEATURE};

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000N7H
fn main() {
    // No given digits
    let mut puzzle = SState::<9, 9, 1, 9>::new();
    let sudoku_constraint = nine_standard_checker();
    let cb = CageBuilder::new(false, &sudoku_constraint);
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
    let cage_constraint = CageChecker::new(cages);
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
    let xsum_constraint = XSumChecker::new(xsums);
    let mut constraint = ConstraintConjunction::new(
        sudoku_constraint,
        ConstraintConjunction::new(cage_constraint, xsum_constraint),);
    let ranker = LinearRanker::new(FeatureVec::from_pairs(vec![
        (NUM_POSSIBLE_FEATURE, -100.0),
        (XSUM_TAIL_FEATURE, 10.0),
        (XSUM_HEAD_FEATURE, 5.0),
        (CAGE_FEATURE, 1.0)
    ]));
    let mut dbg = SamplingDbgObserver::new(0.001);
    let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, false, Some(&mut dbg));
    let maybe_solution = finder.solve().expect("Puzzle solver returned an error:");
    println!("Solution:\n{}", maybe_solution.unwrap().get_state().serialize());
}