use variant_sudoku_dfs::core::FeatureVec;
use variant_sudoku_dfs::ranker::{LinearRanker, NUM_POSSIBLE_FEATURE};
use variant_sudoku_dfs::constraint::ConstraintConjunction;
use variant_sudoku_dfs::solver::{FindFirstSolution, SamplingDbgObserver};
use variant_sudoku_dfs::sudoku::{nine_standard_checker, SState};
use variant_sudoku_dfs::cages::{Cage, CageChecker, CAGE_FEATURE};
use variant_sudoku_dfs::xsums::{XSum, XSumDirection, XSumChecker, XSUM_HEAD_FEATURE, XSUM_TAIL_FEATURE};

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000N7H
fn main() {
    // No given digits
    let mut puzzle = SState::<9, 9, 1, 9>::new();
    let cages = vec![
        Cage { cells: vec![[2, 2], [2, 3]], target: 14, exclusive: false },
        Cage { cells: vec![[2, 7], [3, 7], [4, 7]], target: 15, exclusive: false },
        Cage { cells: vec![[2, 8], [3, 8], [4, 8]], target: 16, exclusive: false },
        Cage { cells: vec![[3, 2], [4, 2], [5, 2]], target: 15, exclusive: false },
        Cage { cells: vec![[3, 3], [4, 3]], target: 12, exclusive: false },
        Cage { cells: vec![[5, 3], [5, 4], [5, 5]], target: 16, exclusive: false },
        Cage { cells: vec![[7, 4], [7, 5]], target: 9, exclusive: false },
        Cage { cells: vec![[8, 2], [8, 3], [8, 4], [8, 5]], target: 18, exclusive: false },
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
    let sudoku_constraint = nine_standard_checker();
    let cage_constraint = CageChecker::new(cages.clone());
    let xsum_constraint = XSumChecker::new(xsums.clone());
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