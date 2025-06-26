use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{LazyLock, Mutex};
use crate::core::{Attribution, ConstraintResult, Error, Feature, Index, Key, RankingInfo, State, Stateful, Unscored, VBitSet, VSetMut, WithId};
use crate::constraint::Constraint;
use crate::sudoku::{stdval_len_bound, stdval_sum_bound, StdOverlay, StdVal};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum XSumDirection {
    // Row, going right (starting from the left)
    RR,
    // Row, going left (starting from the right)
    RL,
    // Column, going down (starting from the top)
    CD,
    // Column, going up (starting from the bottom)
    CU,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct XSum<const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    pub target: u8,
    pub index: usize,
    pub direction: XSumDirection,
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize> Debug for XSum<MIN, MAX, N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "XSum({}, {:?}, {})", self.target, self.index, match self.direction {
            XSumDirection::RR => "RR",
            XSumDirection::RL => "RL",
            XSumDirection::CD => "CD",
            XSumDirection::CU => "CU",
        })
    }
}

pub struct XSumIter<'a, const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    xsum: &'a XSum<MIN, MAX, N, M>,
    r: isize,
    c: isize,
    // Inclusive limit for the iterator
    lim: isize,
}

impl <'a, const MIN: u8, const MAX: u8, const N: usize, const M: usize>
XSumIter<'a, MIN, MAX, N, M> {
    pub fn new(xsum: &'a XSum<MIN, MAX, N, M>, r: usize, c: usize, lim: usize) -> Self {
        XSumIter { xsum, r: r as isize, c: c as isize, lim: lim as isize }
    }
}

impl <'a, const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Iterator for XSumIter<'a, MIN, MAX, N, M> {
    type Item = Index;

    fn next(&mut self) -> Option<Self::Item> {
        if self.r < 0 || self.c < 0 {
            return None;
        }
        let ret = Some([self.r as usize, self.c as usize]);
        match self.xsum.direction {
            XSumDirection::RR => {
                if self.c > self.lim {
                    return None;
                } else {
                    self.c += 1;
                    return ret;
                }
            },
            XSumDirection::RL => {
                if self.c < self.lim {
                    return None;
                } else {
                    self.c -= 1;
                    return ret;
                }
            },
            XSumDirection::CD => {
                if self.r > self.lim {
                    return None;
                } else {
                    self.r += 1;
                    return ret;
                }
            },
            XSumDirection::CU => {
                if self.r < self.lim {
                    return None;
                } else {
                    self.r -= 1;
                    return ret;
                }
            },
        }
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
XSum<MIN, MAX, N, M> {
    pub fn contains_with_len(&self, length: u8, index: Index) -> bool {
        match self.direction {
            XSumDirection::RR => {
                index[0] == self.index && index[1] < length as usize
            },
            XSumDirection::RL => {
                index[0] == self.index && (
                    index[1] as i16 > (M as i16 - 1 - length as i16)
                )
            },
            XSumDirection::CD => {
                index[1] == self.index && index[0] < length as usize
            },
            XSumDirection::CU => {
                index[1] == self.index && (
                    index[0] as i16 > (N as i16 - 1 - length as i16)
                )
            },
        }
    }

    pub fn contains(&self, puzzle: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, index: Index) -> bool {
        let length_index = self.length_index();
        if index == length_index {
            true
        } else if let Some(v) = puzzle.get(length_index) {
            self.contains_with_len(v.val(), index)
        } else {
            false
        }
    }

    pub fn length(&self, puzzle: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>) -> Option<(Index, StdVal<MIN, MAX>)> {
        match self.direction {
            XSumDirection::RR => if let Some(v) = puzzle.get([self.index, 0]) {
                Some(([self.index, 0], v))
            } else {
                None
            },
            XSumDirection::RL => if let Some(v) = puzzle.get([self.index, M - 1]) {
                Some(([self.index, M - 1], v))
            } else {
                None
            },
            XSumDirection::CD => if let Some(v) = puzzle.get([0, self.index]) {
                Some(([0, self.index], v))
            } else {
                None
            },
            XSumDirection::CU => if let Some(v) = puzzle.get([N - 1, self.index]) {
                Some(([N - 1, self.index], v))
            } else {
                None
            },
        }
    }

    pub fn length_index(&self) -> Index {
        match self.direction {
            XSumDirection::RR => [self.index, 0],
            XSumDirection::RL => [self.index, M - 1],
            XSumDirection::CD => [0, self.index],
            XSumDirection::CU => [N - 1, self.index],
        }
    }

    /// Returns an iterator over the range of indices (OTHER than the length
    /// digit) that are part of the XSum. E.g., for an XSum for row index 2
    /// going right with a length digit of 3, the iterator will return (2, 1)
    /// and (2, 2).
    pub fn xrange(&self, length: u8) -> XSumIter<MIN, MAX, N, M> {
        match self.direction {
            XSumDirection::RR => XSumIter::new(self, self.index, 1, std::cmp::min(M, length as usize) - 1),
            XSumDirection::RL => XSumIter::new(self, self.index, M - 2, std::cmp::max(0, (M as i16) - (length as i16)) as usize),
            XSumDirection::CD => XSumIter::new(self, 1, self.index, std::cmp::min(N, length as usize) - 1),
            XSumDirection::CU => XSumIter::new(self, N - 2, self.index, std::cmp::max(0, (N as i16) - (length as i16)) as usize),
        }
    }
}

pub const XSUM_HEAD_FEATURE: &str = "XSUM_HEAD";
pub const XSUM_LN_POSSIBILITIES_FEATURE: &str = "XSUM_LN_POSSIBILITIES";
pub const XSUM_TAIL_FEATURE: &str = "XSUM_TAIL";
pub const XSUM_SUM_OVER_ATTRIBUTION: &str = "XSUM_SUM_OVER";
pub const XSUM_SUM_BAD_ATTRIBUTION: &str = "XSUM_SUM_BAD";
pub const XSUM_SUM_INFEASIBLE_ATTRIBUTION: &str = "XSUM_SUM_INFEASIBLE";
pub const XSUM_LEN_INFEASIBLE_ATTRIBUTION: &str = "XSUM_LEN_INFEASIBLE";

pub struct XSumChecker<const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    xsums: Vec<XSum<MIN, MAX, N, M>>,
    // Remaining to the target
    xsums_remaining: Vec<i16>,
    // Remaining empty NON-LENGTH cells, supposing length is already known
    xsums_empty: Vec<Option<i16>>,
    // To calculate remaining and empty when a length cell becomes known.
    grid: Vec<Option<StdVal<MIN, MAX>>>,
    xsum_head_feature: Key<Feature, WithId>,
    xsum_ln_possibilities_feature: Key<Feature, WithId>,
    xsum_tail_feature: Key<Feature, WithId>,
    xsum_sum_over_attr: Key<Attribution, WithId>,
    xsum_sum_bad_attr: Key<Attribution, WithId>,
    xsum_sum_if_attr: Key<Attribution, WithId>,
    xsum_len_if_attr: Key<Attribution, WithId>,
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize> XSumChecker<MIN, MAX, N, M> {
    pub fn new(xsums: Vec<XSum<MIN, MAX, N, M>>) -> Self {
        let xsums_remaining = xsums.iter().map(|x| x.target as i16).collect();
        let xsums_empty = vec![None; xsums.len()];
        let grid = vec![None; N * M];
        XSumChecker {
            xsums, xsums_remaining, xsums_empty, grid,
            xsum_head_feature: Key::new(XSUM_HEAD_FEATURE).unwrap(),
            xsum_ln_possibilities_feature: Key::new(XSUM_LN_POSSIBILITIES_FEATURE).unwrap(),
            xsum_tail_feature: Key::new(XSUM_TAIL_FEATURE).unwrap(),
            xsum_sum_over_attr: Key::new(XSUM_SUM_OVER_ATTRIBUTION).unwrap(),
            xsum_sum_bad_attr: Key::new(XSUM_SUM_BAD_ATTRIBUTION).unwrap(),
            xsum_sum_if_attr: Key::new(XSUM_SUM_INFEASIBLE_ATTRIBUTION).unwrap(),
            xsum_len_if_attr: Key::new(XSUM_LEN_INFEASIBLE_ATTRIBUTION).unwrap(),
        }
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Debug for XSumChecker<MIN, MAX, N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, x) in self.xsums.iter().enumerate() {
            write!(f, " {:?}\n", x)?;
            write!(f, " - Remaining to target: {}\n", self.xsums_remaining[i])?;
            if let Some(empty) = self.xsums_empty[i] {
                write!(f, " - Empty cells remaining: {}\n", empty)?;
            } else {
                write!(f, " - Length unknown\n")?;
            }
        }
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Stateful<StdVal<MIN, MAX>> for XSumChecker<MIN, MAX, N, M> {
    fn reset(&mut self) {
        self.xsums_remaining = self.xsums.iter().map(|x| x.target as i16).collect();
        self.xsums_empty = vec![None; self.xsums.len()];
        self.grid.fill(None);
    }

    fn apply(&mut self, index: Index, value: StdVal<MIN, MAX>) -> Result<(), Error> {
        self.grid[index[0] * M + index[1]] = Some(value);
        for (i, xsum) in self.xsums.iter().enumerate() {
            let len_index = xsum.length_index();
            if index == len_index {
                let mut remaining = xsum.target as i16 - value.val() as i16;
                let mut empty = value.val() as i16 - 1;
                for i2 in xsum.xrange(value.val()) {
                    if let Some(v) = self.grid[i2[0] * M + i2[1]] {
                        remaining -= v.val() as i16;
                        empty -= 1;
                    }
                }
                self.xsums_remaining[i] = remaining;
                self.xsums_empty[i] = Some(empty);
            } else if let Some(len) = self.grid[len_index[0] * M + len_index[1]] {
                if xsum.contains_with_len(len.val(), index) {
                    self.xsums_remaining[i] -= value.val() as i16;
                    self.xsums_empty[i] = self.xsums_empty[i].map(|e| e - 1);
                }
            }
        }
        Ok(())
    }

    fn undo(&mut self, index: Index, value: StdVal<MIN, MAX>) -> Result<(), Error> {
        self.grid[index[0] * M + index[1]] = None;
        for (i, xsum) in self.xsums.iter().enumerate() {
            let len_index = xsum.length_index();
            if index == len_index {
                // Reset the state for xsum[i]
                self.xsums_remaining[i] = xsum.target as i16;
                self.xsums_empty[i] = None;
            } else if let Some(len) = self.grid[len_index[0] * M + len_index[1]] {
                if xsum.contains_with_len(len.val(), index) {
                    self.xsums_remaining[i] += value.val() as i16;
                    self.xsums_empty[i] = self.xsums_empty[i].map(|e| e + 1);
                }
            }
        }
        Ok(())
    }
}

static ELEM_IN_SUM: LazyLock<Mutex<HashMap<(u8, u8), HashMap<(u8, u8), Option<(u8, u8)>>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});
static XSUM_LENS: LazyLock<Mutex<HashMap<(u8, u8), HashMap<u8, Option<(u8, u8)>>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});
static XSUM_POSSIBILITIES: LazyLock<Mutex<HashMap<(u8, u8), HashMap<u8, usize>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

pub fn elem_in_sum_bound<const MIN: u8, const MAX: u8>(sum: u8, len: u8) -> Option<(u8, u8)> {
    let mut map = ELEM_IN_SUM.lock().unwrap();
    let inner_map = map.entry((MIN, MAX)).or_default();
    if let Some(r) = inner_map.get(&(sum, len)) {
        return *r;
    }
    let r = if len == 0 {
        None
    } else if len == 1 {
        if MIN <= sum && sum <= MAX {
            Some((sum, sum))
        } else {
            None
        }
    } else {
        if let Some((min, max)) = stdval_sum_bound::<MIN, MAX>(len) {
            if !(min <= sum && sum <= max) {
                return None;
            }
        } else {
            return None;
        }
        if let Some((rmin, rmax)) = stdval_sum_bound::<MIN, MAX>(len - 1) {
            let min = if rmax + MIN >= sum {
                MIN
            } else if MIN <= (sum - rmax) && (sum - rmax) <= MAX {
                sum - rmax
            } else {
                return None
            };
            let max = if rmin + MIN > sum {
                return None;
            } else if rmin + MAX <= sum {
                MAX
            } else if MIN <= (sum - rmin) && (sum - rmin) <= MAX {
                sum - rmin
            } else {
                return None;
            };
            Some((min, max))
        } else {
            None
        }
    };
    inner_map.insert((sum, len), r);
    r
}

fn sum_one_out_bound<const MIN: u8, const MAX: u8>(out: u8, len: u8) -> Option<(u8, u8)> {
    assert!(MIN <= out && out <= MAX);
    if MIN + len > MAX {
        return None;
    }
    let mut i = 0;
    let mut j = 0;
    let mut min = 0;
    while i < len {
        let v = MIN + j;
        if v != out {
            min += v;
            i += 1;
        }
        j += 1;
    }
    i = 0;
    j = 0;
    let mut max = 0;
    while i < len {
        let v = MAX - j;
        if v != out {
            max += v;
            i += 1;
        }
        j += 1;
    }
    Some((min, max))
}

pub fn xsum_len_bound<const MIN: u8, const MAX: u8>(sum: u8) -> Option<(u8, u8)> {
    let mut map = XSUM_LENS.lock().unwrap();
    let inner_map = map.entry((MIN, MAX)).or_default();
    if let Some(r) = inner_map.get(&sum) {
        return *r;
    }
    let mut min = None;
    let mut max = None;
    for len in MIN..=MAX {
        if len == 1 {
            if sum == 1 && MIN <= 1 && 1 <= MAX {
                if min.is_none() {
                    min = Some(1);
                }
                max = Some(1);
                break;
            }
            continue;
        } else if len > sum {
            break;
        } else if let Some((smin, smax)) = sum_one_out_bound::<MIN, MAX>(len, len-1) {
            if smin <= sum-len && sum-len <= smax {
                if min.is_none() {
                    min = Some(len);
                }
                max = Some(len)
            }
        }
    }
    let r = if min.is_none() {
        None
    } else {
        Some((min.unwrap(), max.unwrap()))
    };
    inner_map.insert(sum, r);
    r
}

fn n_sum<const MIN: u8, const MAX: u8>(n: u8, sum: u8, remaining: &mut Vec<bool>) -> usize {
    if n == 0 {
        return if sum == 0 { 1 } else { 0 };
    }
    if let Some((lo, hi)) = stdval_len_bound::<MIN, MAX>(sum) {
        if n < lo || hi < n {
            return 0;
        }
        let remaining_vals = remaining
            .iter().enumerate()
            .filter_map(|(v, r)| if *r && (v as u8) + MIN <= sum { Some(v) } else { None })
            .collect::<Vec<_>>();
        let mut total = 0;
        for v in remaining_vals {
            remaining[v] = false;
            total += n_sum::<MIN, MAX>(n-1, sum-MIN-(v as u8), remaining);
            remaining[v] = true;
        }
        total
    } else {
        0
    }
}

pub fn xsum_possibilities<const MIN: u8, const MAX: u8>(sum: u8) -> usize {
    let mut map = XSUM_POSSIBILITIES.lock().unwrap();
    let inner_map = map.entry((MIN, MAX)).or_default();
    if let Some(p) = inner_map.get(&sum) {
        return *p;
    }
    let mut total = 0;
    if let Some((lo, hi)) = xsum_len_bound::<MIN, MAX>(sum) {
        for n in lo..=hi {
            let mut remaining = vec![true; (MAX as usize)+1-(MIN as usize)];
            remaining[(n-1) as usize] = false;
            total += n_sum::<MIN, MAX>(n-1, sum-n, &mut remaining);
        }
    }
    inner_map.insert(sum, total);
    total
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Constraint<StdVal<MIN, MAX>, StdOverlay<N, M>> for XSumChecker<MIN, MAX, N, M> {
    fn check(&self, puzzle: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, ranking: &mut RankingInfo<StdVal<MIN, MAX>, Unscored>) -> ConstraintResult<StdVal<MIN, MAX>> {
        let grid = ranking.cells_mut();
        for (i, xsum) in self.xsums.iter().enumerate() {
            if let Some(e) = self.xsums_empty[i] {
                let r = self.xsums_remaining[i];
                if r < 0 {
                    return ConstraintResult::Contradiction(self.xsum_sum_over_attr);
                } else if e == 0 && r != 0 {
                    return ConstraintResult::Contradiction(self.xsum_sum_bad_attr);
                } else if r == 0 {
                    // Satisfied!
                    continue;
                }
                // We can constrain the empty digits based on the remaining target
                if let Some((min, max)) = elem_in_sum_bound::<MIN, MAX>(r as u8, e as u8) {
                    let mut set = VBitSet::<StdVal<MIN, MAX>>::empty();
                    (min..=max).for_each(|v| set.insert(&StdVal::<MIN, MAX>::new(v)));
                    let len = xsum.length(puzzle).unwrap().1;
                    for i2 in xsum.xrange(len.val()) {
                        let g = &mut grid.get_mut(i2);
                        if puzzle.get(i2).is_none() {
                            g.0.intersect_with(&set);
                        }
                        g.1.add(&self.xsum_tail_feature, 1.0);
                    }
                } else {
                    return ConstraintResult::Contradiction(self.xsum_sum_if_attr);
                }
            } else {
                // We can constrain length based on total target sum
                let len_cell = xsum.length_index();
                if let Some((min, max)) = xsum_len_bound::<MIN, MAX>(xsum.target) {
                    let g = &mut grid.get_mut(len_cell);
                    if puzzle.get(len_cell).is_none() {
                        let mut set = VBitSet::<StdVal<MIN, MAX>>::empty();
                        (min..=max).for_each(|v| set.insert(&StdVal::<MIN, MAX>::new(v)));
                        g.0.intersect_with(&set);
                    }
                    g.1.add(&self.xsum_head_feature, 1.0);
                    let np = xsum_possibilities::<MIN, MAX>(xsum.target) as f64;
                    g.1.add(&self.xsum_ln_possibilities_feature, np.ln_1p())
                } else {
                    return ConstraintResult::Contradiction(self.xsum_len_if_attr);
                }
            }
        }
        ConstraintResult::Ok
    }

    fn debug_at(&self, puzzle: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, index: Index) -> Option<String> {
        let header = "XSumChecker:\n";
        let mut lines = vec![];
        for (i, x) in self.xsums.iter().enumerate() {
            if !x.contains(puzzle, index) {
                continue;
            }
            lines.push(format!("  {:?}", x));
            lines.push(format!("  - Remaining to target: {}", self.xsums_remaining[i]));
            if let Some(empty) = self.xsums_empty[i] {
                lines.push(format!("  - Empty cells remaining: {}", empty));
            } else {
                lines.push(format!("  - Length unknown"));
            }
        }
        if lines.is_empty() {
            None
        } else {
            Some(format!("{}{}", header, lines.join("\n")))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::vec;
    use crate::constraint::test_util::*;
    use crate::core::test_util::replay_givens;
    use crate::solver::test_util::PuzzleReplay;
    use crate::ranker::StdRanker;
    use crate::sudoku::four_standard_parse;

    #[test]
    fn test_elem_in_sum_bound() {
        // {1, 2, 4} is the only valid way to make 7 with 3 digits, so [1, 4] is
        // the range of valid values.
        assert_eq!(elem_in_sum_bound::<1, 9>(7, 3), Some((1, 4)));
        // {6, 8, 9} is the only valid way to make 23 with 3 digits, so [6, 9]
        // is the range of valid values.
        assert_eq!(elem_in_sum_bound::<1, 9>(23, 3), Some((6, 9)));
    }

    #[test]
    fn test_xsum_len_bound() {
        // Possibilities are 2-4, 3-1-2, and 3-2-1, so [2, 3] is the range of
        // valid lens.
        assert_eq!(xsum_len_bound::<1, 9>(6), Some((2, 3)));
        // All the valid possibilities are 8-{permutation of 2..=9}, so [8, 8]
        // is the range of valid lens.
        assert_eq!(xsum_len_bound::<1, 9>(44), Some((8, 8)));
    }

    #[test]
    fn test_xsum_possibilities() {
        // 1 is the only possibility.
        assert_eq!(xsum_possibilities::<1, 9>(1), 1);
        // The length of 1 is too short, the length of 3 is too long, and the
        // length of 2 doesn't work because you can't have two 2s.
        assert_eq!(xsum_possibilities::<1, 9>(4), 0);
        // 2-4, 3-1-2, 3-2-1 are the possibilities.
        assert_eq!(xsum_possibilities::<1, 9>(6), 3);
        // Every possibility is 8-{permutation of 2..=9}
        assert_eq!(xsum_possibilities::<1, 9>(44), 5040 /* == 7! */);
        // Every possibility is 9-{permutation of 1..=8}
        assert_eq!(xsum_possibilities::<1, 9>(45), 40320 /* == 8! */);
    }

    #[test]
    fn test_xsum_contains() {
        let x1 = XSum { direction: XSumDirection::RR, index: 0, target: 5 };
        let x2 = XSum { direction: XSumDirection::RL, index: 0, target: 5 };
        let x3 = XSum { direction: XSumDirection::CD, index: 0, target: 5 };
        let x4 = XSum { direction: XSumDirection::CU, index: 0, target: 5 };
        let mut puzzle1 = four_standard_parse(
            "2..3\n\
             ....\n\
             ....\n\
             4...\n"
        ).unwrap();
        replay_givens(&mut puzzle1);
        // x1 contains two cells -- the length digit and the next
        assert!(x1.contains(&puzzle1, [0, 0]));
        assert!(x1.contains(&puzzle1, [0, 1]));
        assert!(!x1.contains(&puzzle1, [0, 2]));
        // x2 contains three cells -- the length digit and the prev 2
        assert!(x2.contains(&puzzle1, [0, 3]));
        assert!(x2.contains(&puzzle1, [0, 1]));
        assert!(!x2.contains(&puzzle1, [0, 0]));
        // x3 contains two cells -- the length digit and the next
        assert!(x3.contains(&puzzle1, [0, 0]));
        assert!(x3.contains(&puzzle1, [1, 0]));
        assert!(!x3.contains(&puzzle1, [2, 0]));
        // x3 contains four cells -- the length digit and the rest
        assert!(x4.contains(&puzzle1, [3, 0]));
        assert!(x4.contains(&puzzle1, [0, 0]));
        let mut puzzle2 = four_standard_parse(
            "....\n\
             .21.\n\
             .43.\n\
             ....\n"
        ).unwrap();
        replay_givens(&mut puzzle2);
        // These all contain their length cell, but since it's empty, they don't
        // contain any other cells.
        assert!(x1.contains(&puzzle2, [0, 0]));
        assert!(!x1.contains(&puzzle2, [0, 1]));
        assert!(x2.contains(&puzzle2, [0, 3]));
        assert!(!x2.contains(&puzzle2, [0, 2]));
        assert!(x3.contains(&puzzle2, [0, 0]));
        assert!(!x3.contains(&puzzle2, [1, 0]));
        assert!(x4.contains(&puzzle2, [3, 0]));
        assert!(!x4.contains(&puzzle2, [2, 0]));
    }

    #[test]
    fn test_xsum_length() {
        let x1 = XSum { direction: XSumDirection::RR, index: 0, target: 5 };
        let x2 = XSum { direction: XSumDirection::RL, index: 0, target: 5 };
        let x3 = XSum { direction: XSumDirection::CD, index: 0, target: 5 };
        let x4 = XSum { direction: XSumDirection::CU, index: 0, target: 5 };
        let mut puzzle1 = four_standard_parse(
            "2..3\n\
             ....\n\
             ....\n\
             4...\n"
        ).unwrap();
        replay_givens(&mut puzzle1);
        assert_eq!(x1.length(&puzzle1).unwrap(), ([0, 0], StdVal::new(2)));
        assert_eq!(x2.length(&puzzle1).unwrap(), ([0, 3], StdVal::new(3)));
        assert_eq!(x3.length(&puzzle1).unwrap(), ([0, 0], StdVal::new(2)));
        assert_eq!(x4.length(&puzzle1).unwrap(), ([3, 0], StdVal::new(4)));
        let puzzle2 = four_standard_parse(
            "....\n\
             .21.\n\
             .43.\n\
             ....\n"
        ).unwrap();
        assert!(x1.length(&puzzle2).is_none());
        assert!(x2.length(&puzzle2).is_none());
        assert!(x3.length(&puzzle2).is_none());
        assert!(x4.length(&puzzle2).is_none());
    }

    #[test]
    fn test_xsum_xrange() {
        let x1: XSum<1, 4, 4, 4> = XSum { direction: XSumDirection::RR, index: 0, target: 5 };
        let x2: XSum<1, 4, 4, 4> = XSum { direction: XSumDirection::RL, index: 0, target: 5 };
        let x3: XSum<1, 4, 4, 4> = XSum { direction: XSumDirection::CD, index: 0, target: 5 };
        let x4: XSum<1, 4, 4, 4> = XSum { direction: XSumDirection::CU, index: 0, target: 5 };
        // Imagine this puzzle:
        // 2..3
        // ....
        // ....
        // 4...
        assert_eq!(x1.xrange(2).collect::<Vec<_>>(), vec![[0, 1]]);
        assert_eq!(x2.xrange(3).collect::<Vec<_>>(), vec![[0, 2], [0, 1]]);
        assert_eq!(x3.xrange(2).collect::<Vec<_>>(), vec![[1, 0]]);
        assert_eq!(x4.xrange(4).collect::<Vec<_>>(), vec![[2, 0], [1, 0], [0, 0]]);
    }

    #[test]
    fn test_xsum_checker_rr_cd() {
        // XSum 1 is satisfied (2 + 1 = 3).
        let x1 = XSum{ direction: XSumDirection::RR, index: 0, target: 3 };
        // XSum 2 is a failure (3 + 4 + 2 != 10).
        let x2 = XSum{ direction: XSumDirection::CD, index: 2, target: 10 };
        // XSum 3 is a failure even though it's incomplete (after 4, 1 in remaining 3 is infeasible).
        let x3 = XSum{ direction: XSumDirection::RR, index: 2, target: 5 };
        // XSum 4 has no violations because it's incomplete and not over target.
        let x4 = XSum{ direction: XSumDirection::CD, index: 3, target: 10 };
        // XSum 5 has no violations because it doesn't even have a first digit yet.
        let x5 = XSum{ direction: XSumDirection::RR, index: 3, target: 1 };
        // XSum 6 is a failure because there isn't a feasible length.
        let x6 = XSum{ direction: XSumDirection::RR, index: 3, target: 20 };
        // XSum 7 is a failure (3 + 4 > 6)
        let x7 = XSum{ direction: XSumDirection::CD, index: 2, target: 6 };

        let puzzle = four_standard_parse(
            "2134\n\
             ..4.\n\
             432.\n\
             ....\n"
        ).unwrap();

        for (x, expected) in vec![
            (x1, None),
            (x2, Some("XSUM_SUM_BAD")),
            (x3, Some("XSUM_SUM_INFEASIBLE")),
            (x4, None),
            (x5, None),
            (x6, Some("XSUM_LEN_INFEASIBLE")),
            (x7, Some("XSUM_SUM_OVER")),
        ] {
            let mut puzzle = puzzle.clone();
            let ranker = StdRanker::default();
            let mut xsum_checker = XSumChecker::new(vec![x]);
            let result = PuzzleReplay::new(&mut puzzle, &ranker, &mut xsum_checker, None).replay().unwrap();
            if let Some(expected_attr) = expected {
                assert_contradiction(result, expected_attr);
            } else {
                assert_no_contradiction(result);
            }
        }
    }

    #[test]
    fn test_xsum_checker_rl_cu() {
        // XSum 1 is satisfied (2 + 1 = 3).
        let x1 = XSum{ direction: XSumDirection::RL, index: 3, target: 3 };
        // XSum 2 is a failure (3 + 4 + 2 != 10).
        let x2 = XSum{ direction: XSumDirection::CU, index: 1, target: 10 };
        // XSum 3 is a failure even though it's incomplete (4 + 3 + 2 > 5).
        let x3 = XSum{ direction: XSumDirection::RL, index: 1, target: 5 };
        // XSum 4 has no violations because it's incomplete and not over target.
        let x4 = XSum{ direction: XSumDirection::CU, index: 0, target: 10 };
        // XSum 5 has no violations because it doesn't even have a first digit yet.
        let x5 = XSum{ direction: XSumDirection::RL, index: 0, target: 1 };
        // XSum 6 is a failure because there isn't a feasible length.
        let x6 = XSum{ direction: XSumDirection::RL, index: 0, target: 20 };

        let puzzle = four_standard_parse(
            "....\n\
             .234\n\
             .4..\n\
             4312\n"
        ).unwrap();

        for (x, expected) in vec![
            (x1, None),
            (x2, Some("XSUM_SUM_BAD")),
            (x3, Some("XSUM_SUM_OVER")),
            (x4, None),
            (x5, None),
            (x6, Some("XSUM_LEN_INFEASIBLE")),
        ] {
            let mut puzzle = puzzle.clone();
            let ranker = StdRanker::default();
            let mut xsum_checker = XSumChecker::new(vec![x]);
            let result = PuzzleReplay::new(&mut puzzle, &ranker, &mut xsum_checker, None).replay().unwrap();
            if let Some(expected_attr) = expected {
                assert_contradiction(result, expected_attr);
            } else {
                assert_no_contradiction(result);
            }
        }
    }
}