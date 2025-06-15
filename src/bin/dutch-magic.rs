use std::time::Duration;
use variant_sudoku_dfs::core::FeatureVec;
use variant_sudoku_dfs::dutch_whispers::{DutchWhisperBuilder, DutchWhisperChecker};
use variant_sudoku_dfs::magic_squares::{MagicSquare, MagicSquareChecker, MS_FEATURE};
use variant_sudoku_dfs::ranker::{OverlaySensitiveLinearRanker, NUM_POSSIBLE_FEATURE};
use variant_sudoku_dfs::constraint::MultiConstraint;
use variant_sudoku_dfs::solver::PuzzleSetter;
use variant_sudoku_dfs::debug::{DbgObserver, Sample};
use variant_sudoku_dfs::sudoku::{nine_standard_overlay, NineStd, StandardSudokuChecker};
use variant_sudoku_dfs::cages::{CageBuilder, CageChecker, CAGE_FEATURE};
use variant_sudoku_dfs::tui::cli_solve;

// https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000NRF
pub struct DutchMagic;
impl PuzzleSetter<u8, NineStd, OverlaySensitiveLinearRanker, MultiConstraint<u8, NineStd>> for DutchMagic {
    fn setup() -> (NineStd, OverlaySensitiveLinearRanker, MultiConstraint<u8, NineStd>) {
        // No given digits in real puzzle but can be passed in in test.
        Self::setup_with_givens(NineStd::new(nine_standard_overlay()))
    }

    fn setup_with_givens(given: NineStd) -> (NineStd, OverlaySensitiveLinearRanker, MultiConstraint<u8, NineStd>) {
        let puzzle = given;
        let cb = CageBuilder::new(false, puzzle.get_overlay());
        let cages = vec![
            cb.v([2, 4], [3, 4]),
            cb.v([4, 5], [4, 6]),
        ];
        let dw = DutchWhisperBuilder::new(puzzle.get_overlay());
        let whispers = vec![
            dw.row([0, 0], 3),
            dw.row([1, 4], 3),
            dw.row([3, 0], 3),
            dw.row([4, 3], 3),
            dw.row([5, 6], 3),
            dw.row([7, 2], 3),
            dw.row([8, 6], 3),
        ];
        let squares = vec![
            MagicSquare::new([1, 5]),
            MagicSquare::new([3, 1]),
            MagicSquare::new([5, 7]),
            MagicSquare::new([7, 3]),
        ];
        let constraint = MultiConstraint::new(vec_box::vec_box![
            StandardSudokuChecker::new(&puzzle),
            CageChecker::new(cages),
            DutchWhisperChecker::new(whispers),
            MagicSquareChecker::new(squares),
        ]);
        let ranker = OverlaySensitiveLinearRanker::new(FeatureVec::from_pairs(vec![
            (NUM_POSSIBLE_FEATURE, -100.0),
            (CAGE_FEATURE, 1.0),
            (MS_FEATURE, 1.0),
        ]), |_, x, y| x+y);
        (puzzle, ranker, constraint)
    }
}

pub fn main() {
    let mut dbg = DbgObserver::new();
    dbg.sample_print(Sample::every_n(1000))
        .sample_stats("figures/dutch-magic.png", Sample::time(Duration::from_secs(30)));
    cli_solve::<_, _, DutchMagic>(None, dbg);
}

#[cfg(test)]
mod test {
    use variant_sudoku_dfs::{debug::NullObserver, sudoku::nine_standard_parse};
    use super::*;

    #[test]
    fn test_dutch_magic_solution() {
        let input: &str = "195643827\n\
                           627895143\n\
                           438127695\n\
                           951438762\n\
                           276951438\n\
                           843276951\n\
                           514389276\n\
                           769512384\n\
                           38276451.\n";
        let sudoku = nine_standard_parse(input).unwrap();
        let obs = NullObserver;
        cli_solve::<_, _, DutchMagic>(Some(sudoku), obs);
    }
}