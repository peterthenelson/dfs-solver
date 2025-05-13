use dfs_puzzle::dfs::{CompositeStrategy, ConstraintConjunction, FindFirstSolution};
use dfs_puzzle::sudoku::{nine_standard_checker, FirstEmptyStrategy, Sudoku};
use dfs_puzzle::cages::{Cage, CageChecker, CagePartialStrategy};

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000N7H
fn main() {
    // No given digits
    let mut puzzle = Sudoku::<9, 9, 1, 9>::new();
    let cages = vec![
        Cage { cells: vec![(2, 2), (2, 3)], target: 14, exclusive: false },
        Cage { cells: vec![(2, 7), (3, 7), (4, 7)], target: 15, exclusive: false },
        Cage { cells: vec![(2, 8), (3, 8), (4, 8)], target: 16, exclusive: false },
        Cage { cells: vec![(3, 2), (4, 2), (5, 2)], target: 15, exclusive: false },
        Cage { cells: vec![(3, 3), (4, 3)], target: 12, exclusive: false },
        Cage { cells: vec![(5, 3), (5, 4), (5, 5)], target: 16, exclusive: false },
        Cage { cells: vec![(7, 4), (7, 5)], target: 9, exclusive: false },
        Cage { cells: vec![(8, 2), (8, 3), (8, 4), (8, 5)], target: 18, exclusive: false },
    ];
    // TODO: xsums
    let sudoku_constraint = nine_standard_checker();
    let cage_constraint = CageChecker::new(cages.clone());
    let constraint = ConstraintConjunction::new(sudoku_constraint, cage_constraint);
    let cage_strategy = CagePartialStrategy { cages: cages.clone() };
    let strategy = CompositeStrategy::new(
        FirstEmptyStrategy {},
        vec![
            &cage_strategy,
        ],
    );
    let mut finder = FindFirstSolution::new(&mut puzzle, &strategy, &constraint, false);
    let maybe_solution = finder.solve().expect("Puzzle solver returned an error:");
    println!("Solution:\n{}", maybe_solution.unwrap().get_puzzle().serialize());
}