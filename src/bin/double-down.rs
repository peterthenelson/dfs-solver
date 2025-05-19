/*use variant_sudoku_dfs::strategy::CompositeStrategy;
use variant_sudoku_dfs::constraint::ConstraintConjunction;
use variant_sudoku_dfs::solver::FindFirstSolution;
use variant_sudoku_dfs::sudoku::{nine_standard_checker, FirstEmptyStrategy, SState};
use variant_sudoku_dfs::cages::{Cage, CageChecker, CagePartialStrategy};
use variant_sudoku_dfs::xsums::{XSum, XSumDirection, XSumChecker, XSumPartialStrategy};

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
    let constraint = ConstraintConjunction::new(
        sudoku_constraint,
        ConstraintConjunction::new(cage_constraint, xsum_constraint),);
    let cage_strategy = CagePartialStrategy { cages: cages.clone() };
    let xsum_strategy = XSumPartialStrategy { xsums: xsums.clone() };
    let strategy = CompositeStrategy::new(
        FirstEmptyStrategy {},
        vec![
            &xsum_strategy,
            &cage_strategy,
        ],
    );
    let mut finder = FindFirstSolution::new(&mut puzzle, &strategy, &constraint, false);
    let maybe_solution = finder.solve().expect("Puzzle solver returned an error:");
    println!("Solution:\n{}", maybe_solution.unwrap().get_puzzle().serialize());
}
*/
pub fn main() {}