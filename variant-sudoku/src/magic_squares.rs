use std::fmt::Debug;
use crate::{constraint::Constraint, core::{Attribution, CertainDecision, ConstraintResult, Feature, Index, Key, State, Stateful, VBitSet, VSet, VSetMut}, ranker::RankingInfo, sudoku::{NineStdVal, StdOverlay}};

/// This is _standard, exclusive_ magic square. These are extremely limiting--
/// 5 must go in the middle, odds go on the sides, and evens go in the corners,
/// and (of course, by definition), all the different columns/rows/diagonals
/// add up to the same number, which is 15.
#[derive(Debug, Clone)]
pub struct MagicSquare {
    center: Index,
}

impl MagicSquare {
    pub fn new(center: Index) -> Self { Self { center } }

    pub fn contains(&self, index: Index) -> bool {
        if index == self.center {
            return true;
        }
        let dist = std::cmp::max(
            self.center[0].abs_diff(index[0]),
            self.center[1].abs_diff(index[1]),
        );
        dist <= 1
    }

    pub fn row_0(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r - 1, c - 1], [r - 1, c], [r - 1, c + 1]]
    }

    pub fn row_1(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r, c - 1], [r, c], [r, c + 1]]
    }

    pub fn row_2(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r + 1, c - 1], [r + 1, c], [r + 1, c + 1]]
    }

    pub fn col_0(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r - 1, c - 1], [r, c - 1], [r + 1, c - 1]]
    }

    pub fn col_1(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r - 1, c], [r, c], [r + 1, c]]
    }

    pub fn col_2(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r - 1, c + 1], [r, c + 1], [r + 1, c + 1]]
    }

    pub fn diag_0(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r - 1, c - 1], [r, c], [r + 1, c + 1]]
    }

    pub fn diag_1(&self) -> [Index; 3] {
        let [r, c] = self.center;
        [[r - 1, c + 1], [r, c], [r + 1, c - 1]]
    }
}

pub const MS_FEATURE: &str = "MAGIC_SQUARE";
pub const MS_MID_ATTRIBUTION: &str = "MAGIC_SQUARE_MIDDLE_NOT_5";
pub const MS_MID_5_ATTRIBUTION: &str = "MAGIC_SQUARE_MIDDLE_5";
pub const MS_CORNER_ATTRIBUTION: &str = "MAGIC_SQUARE_CORNER_NOT_EVEN";
pub const MS_SIDE_ATTRIBUTION: &str = "MAGIC_SQUARE_SIDE_NOT_ODD";
pub const MS_SUM_ATTRIBUTION: &str = "MAGIC_SQUARE_SUM_NOT_15";
pub const MS_SUM_EXACT_ATTRIBUTION: &str = "MAGIC_SQUARE_SUM_EXACT";
pub const MS_SUM_INFEASIBLE_ATTRIBUTION: &str = "MAGIC_SQUARE_SUM_INFEASIBLE";

pub struct MagicSquareChecker {
    squares: Vec<MagicSquare>,
    evens: VBitSet<NineStdVal>,
    odds: VBitSet<NineStdVal>,
    ms_feature: Key<Feature>,
    ms_mid_attr: Key<Attribution>,
    ms_mid_5_attr: Key<Attribution>,
    ms_corner_attr: Key<Attribution>,
    ms_side_attr: Key<Attribution>,
    ms_sum_attr: Key<Attribution>,
    ms_sum_exact_attr: Key<Attribution>,
    ms_sum_if_attr: Key<Attribution>,
}

impl MagicSquareChecker {
    pub fn new(squares: Vec<MagicSquare>) -> Self {
        let mut evens = VBitSet::<NineStdVal>::empty();
        for v in [2, 4, 6, 8] {
            evens.insert(&NineStdVal::new(v));
        }
        let mut odds = VBitSet::<NineStdVal>::empty();
        for v in [1, 3, 7, 9] {
            odds.insert(&NineStdVal::new(v));
        }
        Self {
            squares,
            evens,
            odds,
            ms_feature: Key::register(MS_FEATURE),
            ms_mid_attr: Key::register(MS_MID_ATTRIBUTION),
            ms_mid_5_attr: Key::register(MS_MID_5_ATTRIBUTION),
            ms_corner_attr: Key::register(MS_CORNER_ATTRIBUTION),
            ms_side_attr: Key::register(MS_SIDE_ATTRIBUTION),
            ms_sum_attr: Key::register(MS_SUM_ATTRIBUTION),
            ms_sum_exact_attr: Key::register(MS_SUM_EXACT_ATTRIBUTION),
            ms_sum_if_attr: Key::register(MS_SUM_INFEASIBLE_ATTRIBUTION),
        }
    }

    fn sum15<const N: usize, const M: usize>(
        &self,
        triple: &[Index; 3], 
        puzzle: &State<NineStdVal, StdOverlay<N, M>>,
        ranking: &RankingInfo<NineStdVal>,
    ) -> Option<ConstraintResult<NineStdVal>> {
        let (sum, n_empty, first_empty) = sum_trip(triple, puzzle);
        if n_empty == 0 {
            if sum != 15 {
                Some(ConstraintResult::Contradiction(self.ms_sum_attr))
            } else {
                None
            }
        } else if n_empty == 1 {
            if sum < 15 {
                let i = first_empty.unwrap();
                let rem = NineStdVal::new(15 - sum);
                if ranking.cells().get(i).0.contains(&rem) {
                    Some(ConstraintResult::Certainty(
                        CertainDecision::new(i, rem),
                        self.ms_sum_exact_attr,
                    ))
                } else {
                    Some(ConstraintResult::Contradiction(self.ms_sum_if_attr))
                }
            } else {
                Some(ConstraintResult::Contradiction(self.ms_sum_attr))
            }
        } else {
            None
        }
    }

}

impl Debug for MagicSquareChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MagicSquareChecker: {:?}\n", self.squares)
    }
}

impl Stateful<NineStdVal> for MagicSquareChecker {}

fn check_vals<const N: usize, const M: usize>(
    indices: &[Index; 4],
    values: &VBitSet<NineStdVal>,
    puzzle: &State<NineStdVal, StdOverlay<N, M>>,
    ranking: &mut RankingInfo<NineStdVal>,
    attribution: Key<Attribution>,
) -> Option<ConstraintResult<NineStdVal>> {
    let grid = ranking.cells_mut();
    for i in indices {
        if let Some(v) = puzzle.get(*i) {
            if !values.contains(&v) {
                return Some(ConstraintResult::Contradiction(attribution))
            }
        } else {
            grid.get_mut(*i).0.intersect_with(values);
        }
    }
    None
}

fn sum_trip<const N: usize, const M: usize>(
    triple: &[Index; 3], 
    puzzle: &State<NineStdVal, StdOverlay<N, M>>,
) -> (u8, u8, Option<Index>) {
    let mut sum = 0;
    let mut n_empty = 3;
    let mut first_empty = None;
    for i in triple {
        if let Some(v) = puzzle.get(*i) {
            sum += v.val();
            n_empty -= 1;
        } else if first_empty.is_none() {
            first_empty = Some(*i)
        }
    }
    (sum, n_empty, first_empty)
}

impl <const N: usize, const M: usize>
Constraint<NineStdVal, StdOverlay<N, M>> for MagicSquareChecker {
    fn check(&self, puzzle: &State<NineStdVal, StdOverlay<N, M>>, ranking: &mut RankingInfo<NineStdVal>) -> ConstraintResult<NineStdVal> {
        for square in &self.squares {
            if let Some(v) = puzzle.get(square.center) {
                if v.val() != 5 {
                    return ConstraintResult::Contradiction(self.ms_mid_attr);
                }
            } else {
                return ConstraintResult::Certainty(CertainDecision::new(square.center, NineStdVal::new(5)), self.ms_mid_5_attr);
            }
            let [ul, _, lr] = square.diag_0();
            let [ll, _, ur] = square.diag_1();
            if let Some(cr) = check_vals(&[ul, lr, ll, ur], &self.evens, puzzle, ranking, self.ms_corner_attr) {
                return cr;
            }
            let [ml, _, mr] = square.row_1();
            let [mu, _, md] = square.col_1();
            if let Some(cr) = check_vals(&[ml, mr, mu, md], &self.odds, puzzle, ranking, self.ms_side_attr) {
                return cr;
            }
            for triple in [
                square.col_0(),
                square.col_1(),
                square.col_2(),
                square.row_0(),
                square.row_1(),
                square.row_2(),
                square.diag_0(),
                square.diag_1(),
            ] {
                if let Some(cr) = self.sum15(&triple, puzzle, ranking) {
                    return cr;
                }
            }
            for r in 0..3 {
                for c in 0..3 {
                    let i = [square.center[0] - 1 + r, square.center[1] - 1 + c];
                    ranking.cells_mut().get_mut(i).1.add(&self.ms_feature, 1.0);
                }
            }
        }
        ConstraintResult::Ok
    }

    fn debug_at(&self, _: &State<NineStdVal, StdOverlay<N, M>>, index: Index) -> Option<String> {
        for square in &self.squares {
            let [ul, mm, lr] = square.diag_0();
            let [ll, _, ur] = square.diag_1();
            let [ml, _, mr] = square.row_1();
            let [mu, _, md] = square.col_1();
            if index == ul || index == lr || index == ll || index == ur {
                return Some("Magic Square: corner cell (must be even)".to_string());
            } else if index == ml || index == mr || index == mu || index == md {
                return Some("Magic Square: side cell (must be odd)".to_string());
            } else if index == mm {
                return Some("Magic Square: middle cell (must be 5)".to_string())
            }
        }
        None
    }
}

#[cfg(test)]
mod test {
    use crate::{constraint::test_util::{assert_contradiction, assert_no_contradiction}, magic_squares::{MagicSquare, MagicSquareChecker}, ranker::StdRanker, solver::test_util::PuzzleReplay, sudoku::nine_standard_parse};

    #[test]
    fn test_magic_square_checker() {
        // Each magic square is a box in the 9x9 grid
        // #1 is satisfied
        let ms1 = MagicSquare::new([1, 1]);
        // #2 has the wrong center digit
        let ms2 = MagicSquare::new([1, 4]);
        // #3 has an odd digit in the corner
        let ms3 = MagicSquare::new([1, 7]);
        // #4 has an even digit on the side 
        let ms4 = MagicSquare::new([4, 1]);
        // #5 has a bad sum (2+5+6 != 15)
        let ms5 = MagicSquare::new([4, 4]);
        // #6 has an infeasible sum (15-2-8=5, which is not avail)
        let ms6 = MagicSquare::new([4, 7]);

        let puzzle = nine_standard_parse(
            "2 9 4|. . .|1 . .\n\
             7 5 3|. 6 .|. 5 .\n\
             6 1 8|. . .|. . .\n\
             -----+-----+-----\n\
             . . .|2 . .|2 . .\n\
             2 5 .|. 5 .|. 5 .\n\
             . . .|. . 6|8 . .\n\
             -----+-----+-----\n\
             . . .|. . .|. . .\n\
             . . .|. . .|. . .\n\
             . . .|. . .|. . .\n"
        ).unwrap();

        for (ms, expected) in vec![
            (ms1, None),
            (ms2, Some("MAGIC_SQUARE_MIDDLE_NOT_5")),
            (ms3, Some("MAGIC_SQUARE_CORNER_NOT_EVEN")),
            (ms4, Some("MAGIC_SQUARE_SIDE_NOT_ODD")),
            (ms5, Some("MAGIC_SQUARE_SUM_NOT_15")),
            (ms6, Some("MAGIC_SQUARE_SUM_INFEASIBLE")),
        ] {
            let mut puzzle = puzzle.clone();
            let ranker = StdRanker::default();
            let mut ms_checker = MagicSquareChecker::new(vec![ms]);
            let result = PuzzleReplay::new(&mut puzzle, &ranker, &mut ms_checker, None).replay().unwrap();
            if let Some(attr) = expected {
                assert_contradiction(result, attr);
            } else {
                assert_no_contradiction(result);
            }
        }
    }
}