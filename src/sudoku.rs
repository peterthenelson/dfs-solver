use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::sync::{LazyLock, Mutex};
use crate::core::{full_set, to_value, unpack_values, ConstraintResult, DecisionGrid, Error, Index, Set, State, Stateful, UVGrid, UVUnwrapped, UVWrapped, UVal, Value};
use crate::constraint::{Constraint, ConstraintConjunction, ConstraintViolationDetail};

/// Standard Sudoku value, ranging from a minimum to a maximum value (inclusive).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SVal<const MIN: u8, const MAX: u8>(u8);

impl <const MIN: u8, const MAX: u8> SVal<MIN, MAX> {
    pub fn new(value: u8) -> Self {
        assert!(value >= MIN && value <= MAX, "Value out of bounds");
        SVal(value)
    }

    pub fn val(self) -> u8 {
        self.0
    }
}

impl <const MIN: u8, const MAX: u8> Value<u8> for SVal<MIN, MAX> {
    fn parse(s: &str) -> Result<Self, Error> {
        let value = s.parse::<u8>().map_err(|v| Error::new(format!("Invalid value: {}", v).to_string()))?;
        if value < MIN || value > MAX {
            return Err(Error::new(format!("Value out of bounds: {} ({}-{})", value, MIN, MAX)));
        }
        Ok(SVal(value))
    }

    fn cardinality() -> usize {
        (MAX - MIN + 1) as usize
    }

    fn possiblities() -> Vec<Self> {
        (MIN..=MAX).map(|v| SVal(v)).collect()
    }

    fn from_uval(u: UVal<u8, UVUnwrapped>) -> Self {
        SVal(u.value() + MIN)
    }

    fn to_uval(self) -> UVal<u8, UVWrapped> {
        UVal::new(self.0 - MIN)
    }
}

pub fn unpack_sval_vals<const MIN: u8, const MAX: u8>(s: &Set<u8>) -> Vec<u8> {
    unpack_values::<u8, SVal<MIN, MAX>>(&s).iter().map(|v| v.val()).collect::<Vec<u8>>()
}

/// Tables of useful sums in Sudoku.
static SVAL_LEN_TO_SUM: LazyLock<Mutex<HashMap<(u8, u8), HashMap<u8, Option<(u8, u8)>>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});
static SVAL_SUM_TO_LEN: LazyLock<Mutex<HashMap<(u8, u8), HashMap<u8, Option<(u8, u8)>>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

// Useful utility for working with sums of ranges of numbers. Returns the range
// of sums that are possible to from len exclusive numbers drawn from the range
// min..=max.
pub fn min_max_sum(min: u8, max: u8, len: u8) -> Option<(u8, u8)> {
    // TODO: Use n(n+1)/2 formula instead of loops
    if min + len - 1 > max {
        None
    } else {
        Some(((min..=(min+len-1)).sum(), ((max+1-len)..=max).sum()))
    }
}

// Not efficient, but hey it's subset sum, so what are you going to do.
fn sum_feasible(min: u8, max: u8, k: u8, sum: u8) -> bool {
    if k == 0 {
        sum == 0
    } else if k == 1 {
        min <= sum && sum <= max
    } else if min >= max {
        false
    } else if sum < min {
        false
    } else if sum_feasible(min+1, max, k-1, sum-min) {
        true
    } else {
        sum_feasible(min+1, max, k, sum)
    }
}

// What is the minimum/maximum sum that len exclusive SVal<MIN, MAX>s could add up to?
pub fn sval_sum_bound<const MIN: u8, const MAX: u8>(len: u8) -> Option<(u8, u8)> {
    let mut map = SVAL_LEN_TO_SUM.lock().unwrap();
    let inner_map = map.entry((MIN, MAX)).or_default();
    if let Some(r) = inner_map.get(&len) {
        return *r
    }
    let r = min_max_sum(MIN, MAX, len);
    inner_map.insert(len, r);
    r
}

// What is the minimum/maximum len of exclusive SVal<MIN, MAX>s that could add up to sum?
pub fn sval_len_bound<const MIN: u8, const MAX: u8>(sum: u8) -> Option<(u8, u8)> {
    let mut map = SVAL_SUM_TO_LEN.lock().unwrap();
    let inner_map = map.entry((MIN, MAX)).or_default();
    if let Some(r) = inner_map.get(&sum) {
        return *r
    }
    let mut min = None;
    let mut max = None;
    for len in 1..u8::MAX {
        let range = min_max_sum(MIN, MAX, len);
        if let Some((lo, hi)) = range {
            if lo <= sum && sum <= hi && sum_feasible(MIN, MAX, len, sum) {
                if min.is_none() {
                    min = Some(len);
                }
                max = Some(len);
            } else if lo > sum {
                break;
            }
        }
    }
    let r = if min.is_none() { None } else { Some((min.unwrap(), max.unwrap())) };
    inner_map.insert(sum, r);
    r
}

/// Standard rectangular Sudoku grid.
#[derive(Clone)]
pub struct SState<const N: usize, const M: usize, const MIN: u8, const MAX: u8> {
    grid: UVGrid<u8>,
} 

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Debug for SState<N, M, MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.serialize())
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> SState<N, M, MIN, MAX> {
    pub fn new() -> Self {
        Self { grid: UVGrid::new(N, M) }
    }

    pub fn parse(s: &str) -> Result<Self, Error> {
        let mut grid = UVGrid::new(N, M);
        let lines: Vec<&str> = s.lines().collect();
        if lines.len() != N {
            return Err(Error::new("Invalid number of rows".to_string()));
        }
        for i in 0..N {
            let line = lines[i].trim();
            if line.len() != M {
                return Err(Error::new("Invalid number of columns".to_string()));
            }
            for j in 0..M {
                let c = line.chars().nth(j).unwrap();
                if c == '.' {
                    // Already None
                } else {
                    let s = c.to_string();
                    let v = SVal::<MIN, MAX>::parse(s.as_str())?;
                    grid.set([i, j], Some(v.to_uval()));
                }
            }
        }
        Ok(Self { grid })
    }

    pub fn serialize(&self) -> String {
        let mut result = String::new();
        for r in 0..N {
            for c in 0..M {
                if let Some(v) = self.grid.get([r, c]) {
                    result.push_str(to_value::<u8, SVal<MIN, MAX>>(v).val().to_string().as_str());
                } else {
                    result.push('.');
                }
            }
            result.push('\n');
        }
        result
    }
}

pub fn nine_standard_parse(s: &str) -> Result<SState<9, 9, 1, 9>, Error> {
    SState::<9, 9, 1, 9>::parse(s)
}

pub fn eight_standard_parse(s: &str) -> Result<SState<8, 8, 1, 8>, Error> {
    SState::<8, 8, 1, 8>::parse(s)
}

pub fn six_standard_parse(s: &str) -> Result<SState<6, 6, 1, 6>, Error> {
    SState::<6, 6, 1, 6>::parse(s)
}

pub fn four_standard_parse(s: &str) -> Result<SState<4, 4, 1, 4>, Error> {
    SState::<4, 4, 1, 4>::parse(s)
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Display
for SState<N, M, MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        for r in 0..N {
            for c in 0..M {
                if let Some(v) = self.grid.get([r, c]) {
                    write!(f, "{}", to_value::<u8, SVal::<MIN, MAX>>(v).val())?;
                } else {
                    write!(f, ".")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub const OUT_OF_BOUNDS_ERROR: Error = Error::new_const("Out of bounds");
pub const ALREADY_FILLED_ERROR: Error = Error::new_const("Cell already filled");
pub const NO_SUCH_ACTION_ERROR: Error = Error::new_const("No such action to undo");
pub const UNDO_MISMATCH: Error = Error::new_const("Undo value mismatch");
pub const ILLEGAL_ACTION_RC: Error = Error::new_const("A row/col violation already exists; can't apply further actions.");
pub const ILLEGAL_ACTION_BOX: Error = Error::new_const("A box violation already exists; can't apply further actions.");

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> State<u8> for SState<N, M, MIN, MAX> {
    type Value = SVal<MIN, MAX>;
    const ROWS: usize = N;
    const COLS: usize = M;
    fn get(&self, index: Index) -> Option<Self::Value> {
        if index[0] >= N || index[1] >= M {
            return None;
        }
        self.grid.get(index).map(to_value)
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Stateful<u8, SVal<MIN, MAX>> for SState<N, M, MIN, MAX> {
    fn reset(&mut self) {
        self.grid = UVGrid::new(N, M);
    }

    fn apply(&mut self, index: Index, value: SVal<MIN, MAX>) -> Result<(), Error> {
        if index[0] >= N || index[1] >= M {
            return Err(OUT_OF_BOUNDS_ERROR);
        }
        if self.grid.get(index).is_some() {
            return Err(ALREADY_FILLED_ERROR);
        }
        self.grid.set(index, Some(value.to_uval()));
        Ok(())
    }

    fn undo(&mut self, index: Index, value: SVal<MIN, MAX>) -> Result<(), Error> {
        if index[0] >= N || index[1] >= M {
            return Err(OUT_OF_BOUNDS_ERROR);
        }
        match self.grid.get(index) {
            None => return Err(NO_SUCH_ACTION_ERROR),
            Some(v) => {
                if v != value.to_uval() {
                    return Err(UNDO_MISMATCH);
                }
            }
        }
        self.grid.set(index, None);
        Ok(())
    }
}

// For components outside of the sudoku module that may need to borrow these
// constraints' notion of visibility.
pub trait VisibilityPartition {
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool;

    fn all_mutually_visible(&self, indices: &Vec<Index>) -> bool {
        indices.iter().all(|i| self.mutually_visible(indices[0], *i))
    }
}

pub struct RowColChecker<const N: usize, const M: usize, const MIN: u8, const MAX: u8> {
    row: [Set<u8>; N],
    col: [Set<u8>; M],
    illegal: Option<(Index, SVal<MIN, MAX>)>,
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8>
VisibilityPartition for RowColChecker<N, M, MIN, MAX> {
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        i1[0] == i2[0] || i1[1] == i2[1]
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> RowColChecker<N, M, MIN, MAX> {
    pub fn new() -> Self {
        return RowColChecker {
            row: std::array::from_fn(|_| full_set::<u8, SVal<MIN, MAX>>()),
            col: std::array::from_fn(|_| full_set::<u8, SVal<MIN, MAX>>()),
            illegal: None,
        }
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Debug for RowColChecker<N, M, MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((i, v)) = self.illegal {
            write!(f, "Illegal move: {:?}; {:?}\n", i, v)?;
        }
        write!(f, "Unused vals by row:\n")?;
        for r in 0..N {
            let vals = unpack_sval_vals::<MIN, MAX>(&self.row[r]);
            write!(f, " {}: {:?}\n", r, vals)?;
        }
        write!(f, "Unused vals by col:\n")?;
        for c in 0..M {
            let vals = unpack_sval_vals::<MIN, MAX>(&self.col[c]);
            write!(f, " {}: {:?}\n", c, vals)?;
        }
        Ok(())
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Stateful<u8, SVal<MIN, MAX>> for RowColChecker<N, M, MIN, MAX> {
    fn reset(&mut self) {
        self.row = std::array::from_fn(|_| full_set::<u8, SVal<MIN, MAX>>());
        self.col = std::array::from_fn(|_| full_set::<u8, SVal<MIN, MAX>>());
        self.illegal = None;
    }

    fn apply(&mut self, index: Index, value: SVal<MIN, MAX>) -> Result<(), Error> {
        let uv = value.to_uval();
        // In theory we could be allow multiple illegal moves and just
        // invalidate and recalculate the grid or something, but it seems hard.
        if self.illegal.is_some() {
            return Err(ILLEGAL_ACTION_RC);
        }
        if !self.row[index[0]].contains(uv) || !self.col[index[1]].contains(uv) {
            self.illegal = Some((index, value));
            return Ok(());
        }
        self.row[index[0]].remove(uv);
        self.col[index[1]].remove(uv);
        Ok(())
    }

    fn undo(&mut self, index: Index, value: SVal<MIN, MAX>) -> Result<(), Error> {
        if let Some((i, v)) = self.illegal {
            if i != index || v != value {
                return Err(UNDO_MISMATCH);
            } else {
                self.illegal = None;
                return Ok(());
            }
        }
        let uv = value.to_uval();
        self.row[index[0]].insert(uv);
        self.col[index[1]].insert(uv);
        Ok(())
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> Constraint<u8, SState<N, M, MIN, MAX>> for RowColChecker<N, M, MIN, MAX> {
    fn check(&self, _: &SState<N, M, MIN, MAX>, grid: &mut DecisionGrid<u8, SVal<MIN, MAX>>) -> ConstraintResult<u8, SVal<MIN, MAX>> {
        if self.illegal.is_some() {
            return ConstraintResult::Contradiction;
        }
        for r in 0..N {
            for c in 0..M {
                let cell = grid.get_mut([r, c]);
                cell.0.intersect_with(&self.row[r]);
                cell.0.intersect_with(&self.col[c]);
            }
        }
        ConstraintResult::Ok
    }

    fn explain_contradictions(&self, _: &SState<N, M, MIN, MAX>) -> Vec<ConstraintViolationDetail> {
        todo!()
    }
}

pub struct BoxChecker<const N: usize, const M: usize, const MIN: u8, const MAX: u8> {
    br: usize,
    bc: usize,
    bh: usize,
    bw: usize,
    boxes: Box<[Set<u8>]>,
    illegal: Option<(Index, SVal<MIN, MAX>)>,
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> BoxChecker<N, M, MIN, MAX> {
    pub fn new(br: usize, bc: usize, bh: usize, bw: usize) -> Self {
        if N != br*bh {
            panic!("BoxChecker expected N == br*bh, but {} != {}*{}", N, br, bh);
        } else if M != bc*bw {
            panic!("BoxChecker expected M == bc*bw, but {} != {}*{}", M, bc, bw);
        }
        Self {
            br, bc, bh, bw,
            boxes: vec![full_set::<u8, SVal<MIN, MAX>>(); br * bc].into_boxed_slice(),
            illegal: None,
        }
    }

    pub fn box_coords(&self, index: Index) -> Index {
        [index[0] / self.bh, index[1] / self.bw]
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8>
VisibilityPartition for BoxChecker<N, M, MIN, MAX> {
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        self.box_coords(i1) == self.box_coords(i2)
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8>
Debug for BoxChecker<N, M, MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((i, v)) = self.illegal {
            write!(f, "Illegal move: {:?}; {:?}\n", i, v)?;
        }
        write!(f, "Unused vals by box:\n")?;
        for r in 0..self.br {
            for c in 0..self.bc {
                let vals = unpack_values::<u8, SVal<MIN, MAX>>(&self.boxes[r*self.bh+c]).iter().map(|v| v.val()).collect::<Vec<u8>>();
                write!(f, " {},{}: {:?}\n", r, c, vals)?;
            }
        }
        Ok(())
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8>
Stateful<u8, SVal<MIN, MAX>> for BoxChecker<N, M, MIN, MAX> {
    fn reset(&mut self) {
        self.boxes = vec![full_set::<u8, SVal<MIN, MAX>>(); self.br * self.bc].into_boxed_slice();
        self.illegal = None;
    }

    fn apply(&mut self, index: Index, value: SVal<MIN, MAX>) -> Result<(), Error> {
        let uv = value.to_uval();
        // In theory we could be allow multiple illegal moves and just
        // invalidate and recalculate the grid or something, but it seems hard.
        if self.illegal.is_some() {
            return Err(ILLEGAL_ACTION_BOX);
        }
        let bindex = self.box_coords(index);
        if !self.boxes[bindex[0]*self.bh+bindex[1]].contains(uv) {
            self.illegal = Some((index, value));
            return Ok(());
        }
        self.boxes[bindex[0]*self.bh+bindex[1]].remove(uv);
        Ok(())
    }

    fn undo(&mut self, index: Index, value: SVal<MIN, MAX>) -> Result<(), Error> {
        if let Some((i, v)) = self.illegal {
            if i != index || v != value {
                return Err(UNDO_MISMATCH);
            } else {
                self.illegal = None;
                return Ok(());
            }
        }
        let uv = value.to_uval();
        let bindex = self.box_coords(index);
        self.boxes[bindex[0]*self.bh+bindex[1]].insert(uv);
        Ok(())
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8>
Constraint<u8, SState<N, M, MIN, MAX>> for BoxChecker<N, M, MIN, MAX> {
    fn check(&self, _: &SState<N, M, MIN, MAX>, grid: &mut DecisionGrid<u8, SVal<MIN, MAX>>) -> ConstraintResult<u8, SVal<MIN, MAX>> {
        if self.illegal.is_some() {
            return ConstraintResult::Contradiction;
        }
        for r in 0..N {
            for c in 0..M {
                let bindex = self.box_coords([r, c]);
                let cell = grid.get_mut([r, c]);
                cell.0.intersect_with(&self.boxes[bindex[0]*self.bh + bindex[1]]);
            }
        }
        ConstraintResult::Ok
    }

    fn explain_contradictions(&self, _: &SState<N, M, MIN, MAX>) -> Vec<ConstraintViolationDetail> {
        todo!()
    }
}

pub struct NineBoxChecker(BoxChecker<9, 9, 1, 9>);
impl NineBoxChecker {
    pub fn new() -> Self {
        NineBoxChecker(BoxChecker::new(3, 3, 3, 3))
    }
}
impl Debug for NineBoxChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl VisibilityPartition for NineBoxChecker {
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        self.0.mutually_visible(i1, i2)
    }
}
impl Stateful<u8, SVal<1, 9>> for NineBoxChecker {
    fn reset(&mut self) {
        self.0.reset()
    }
    fn apply(&mut self, index: Index, value: SVal<1, 9>) -> Result<(), Error> {
        self.0.apply(index, value)
    }
    fn undo(&mut self, index: Index, value: SVal<1, 9>) -> Result<(), Error> {
        self.0.undo(index, value)
    }
}
impl Constraint<u8, SState<9, 9, 1, 9>> for NineBoxChecker {
    fn check(&self, puzzle: &SState<9, 9, 1, 9>, grid: &mut DecisionGrid<u8, SVal<1, 9>>) -> ConstraintResult<u8, SVal<1, 9>> {
        self.0.check(puzzle, grid)
    }
    fn explain_contradictions(&self, puzzle: &SState<9, 9, 1, 9>) -> Vec<ConstraintViolationDetail> {
        self.0.explain_contradictions(puzzle)
    }
}
pub type NineStandardChecker = ConstraintConjunction<u8, SState<9, 9, 1, 9>, RowColChecker<9, 9, 1, 9>, NineBoxChecker>;
impl VisibilityPartition for NineStandardChecker {
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        self.x.mutually_visible(i1, i2) || self.y.mutually_visible(i1, i2)
    }
}
pub fn nine_standard_checker() -> NineStandardChecker {
    NineStandardChecker::new(RowColChecker::new(), NineBoxChecker::new())
}

pub struct EightBoxChecker(BoxChecker<8, 8, 1, 8>);
impl EightBoxChecker {
    pub fn new() -> Self {
        EightBoxChecker(BoxChecker::new(4, 2, 2, 4))
    }
}
impl Debug for EightBoxChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl VisibilityPartition for EightBoxChecker {
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        self.0.mutually_visible(i1, i2)
    }
}
impl Stateful<u8, SVal<1, 8>> for EightBoxChecker {
    fn reset(&mut self) {
        self.0.reset()
    }
    fn apply(&mut self, index: Index, value: SVal<1, 8>) -> Result<(), Error> {
        self.0.apply(index, value)
    }
    fn undo(&mut self, index: Index, value: SVal<1, 8>) -> Result<(), Error> {
        self.0.undo(index, value)
    }
}
impl Constraint<u8, SState<8, 8, 1, 8>> for EightBoxChecker {
    fn check(&self, puzzle: &SState<8, 8, 1, 8>, grid: &mut DecisionGrid<u8, SVal<1, 8>>) -> ConstraintResult<u8, SVal<1, 8>> {
        self.0.check(puzzle, grid)
    }
    fn explain_contradictions(&self, puzzle: &SState<8, 8, 1, 8>) -> Vec<ConstraintViolationDetail> {
        self.0.explain_contradictions(puzzle)
    }
}
pub type EightStandardChecker = ConstraintConjunction<u8, SState<8, 8, 1, 8>, RowColChecker<8, 8, 1, 8>, EightBoxChecker>;
impl VisibilityPartition for EightStandardChecker {
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        self.x.mutually_visible(i1, i2) || self.y.mutually_visible(i1, i2)
    }
}
pub fn eight_standard_checker() -> EightStandardChecker {
    EightStandardChecker::new(RowColChecker::new(), EightBoxChecker::new())
}

pub struct SixBoxChecker(BoxChecker<6, 6, 1, 6>);
impl SixBoxChecker {
    pub fn new() -> Self {
        SixBoxChecker(BoxChecker::new(3, 2, 2, 3))
    }
}
impl Debug for SixBoxChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl VisibilityPartition for SixBoxChecker {
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        self.0.mutually_visible(i1, i2)
    }
}
impl Stateful<u8, SVal<1, 6>> for SixBoxChecker {
    fn reset(&mut self) {
        self.0.reset()
    }
    fn apply(&mut self, index: Index, value: SVal<1, 6>) -> Result<(), Error> {
        self.0.apply(index, value)
    }
    fn undo(&mut self, index: Index, value: SVal<1, 6>) -> Result<(), Error> {
        self.0.undo(index, value)
    }
}
impl Constraint<u8, SState<6, 6, 1, 6>> for SixBoxChecker {
    fn check(&self, puzzle: &SState<6, 6, 1, 6>, grid: &mut DecisionGrid<u8, SVal<1, 6>>) -> ConstraintResult<u8, SVal<1, 6>> {
        self.0.check(puzzle, grid)
    }
    fn explain_contradictions(&self, puzzle: &SState<6, 6, 1, 6>) -> Vec<ConstraintViolationDetail> {
        self.0.explain_contradictions(puzzle)
    }
}
pub type SixStandardChecker = ConstraintConjunction<u8, SState<6, 6, 1, 6>, RowColChecker<6, 6, 1, 6>, SixBoxChecker>;
impl VisibilityPartition for SixStandardChecker {
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        self.x.mutually_visible(i1, i2) || self.y.mutually_visible(i1, i2)
    }
}
pub fn six_standard_checker() -> SixStandardChecker {
    SixStandardChecker::new(RowColChecker::new(), SixBoxChecker::new())
}

pub struct FourBoxChecker(BoxChecker<4, 4, 1, 4>);
impl FourBoxChecker {
    pub fn new() -> Self {
        FourBoxChecker(BoxChecker::new(2, 2, 2, 2))
    }
}
impl Debug for FourBoxChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl VisibilityPartition for FourBoxChecker {
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        self.0.mutually_visible(i1, i2)
    }
}
impl Stateful<u8, SVal<1, 4>> for FourBoxChecker {
    fn reset(&mut self) {
        self.0.reset()
    }
    fn apply(&mut self, index: Index, value: SVal<1, 4>) -> Result<(), Error> {
        self.0.apply(index, value)
    }
    fn undo(&mut self, index: Index, value: SVal<1, 4>) -> Result<(), Error> {
        self.0.undo(index, value)
    }
}
impl Constraint<u8, SState<4, 4, 1, 4>> for FourBoxChecker {
    fn check(&self, puzzle: &SState<4, 4, 1, 4>, grid: &mut DecisionGrid<u8, SVal<1, 4>>) -> ConstraintResult<u8, SVal<1, 4>> {
        self.0.check(puzzle, grid)
    }
    fn explain_contradictions(&self, puzzle: &SState<4, 4, 1, 4>) -> Vec<ConstraintViolationDetail> {
        self.0.explain_contradictions(puzzle)
    }
}
pub type FourStandardChecker = ConstraintConjunction<u8, SState<4, 4, 1, 4>, RowColChecker<4, 4, 1, 4>, FourBoxChecker>;
impl VisibilityPartition for FourStandardChecker {
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        self.x.mutually_visible(i1, i2) || self.y.mutually_visible(i1, i2)
    }
}
pub fn four_standard_checker() -> FourStandardChecker {
    FourStandardChecker::new(RowColChecker::new(), FourBoxChecker::new())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{core::{empty_set, UInt}, ranker::LinearRanker, solver::FindFirstSolution};
    use crate::core::test_util::round_trip_value;

    #[test]
    fn test_sval() {
        // Closed interval, so it's 9-3+1
        assert_eq!(SVal::<3, 9>::cardinality(), 7);
        // Values get serialized in the range 0..=(MAX-MIN)
        assert_eq!(SVal::<3, 9>(3).to_uval(), UVal::new(0));
        assert_eq!(SVal::<3, 9>(6).to_uval(), UVal::new(3));
        assert_eq!(SVal::<3, 9>(9).to_uval(), UVal::new(6));
        // This round-trips
        assert_eq!(round_trip_value(SVal::<3, 9>(3)).val(), 3);
        assert_eq!(round_trip_value(SVal::<3, 9>(6)).val(), 6);
        assert_eq!(round_trip_value(SVal::<3, 9>(9)).val(), 9);
    }

    #[test]
    fn test_sval_set() {
        let mut mostly_empty = empty_set::<u8, SVal<3, 9>>();
        assert_eq!(unpack_sval_vals::<3, 9>(&mostly_empty), vec![]);
        mostly_empty.insert(SVal::<3, 9>::new(4).to_uval());
        assert_eq!(unpack_sval_vals::<3, 9>(&mostly_empty), vec![4]);
        let mut mostly_full = full_set::<u8, SVal<3, 9>>();
        assert_eq!(
            unpack_sval_vals::<3, 9>(&mostly_full),
            vec![3, 4, 5, 6, 7, 8, 9],
        );
        mostly_full.remove(SVal::<3, 9>::new(4).to_uval());
        assert_eq!(
            unpack_sval_vals::<3, 9>(&mostly_full),
            vec![3, 5, 6, 7, 8, 9],
        );
    }

    #[test]
    fn test_sval_stats() {
        // Min=1, Max=4
        assert_eq!(sval_sum_bound::<1, 4>(1), Some((1, 4)));
        // Min=1+2, Max=4+3
        assert_eq!(sval_sum_bound::<1, 4>(2), Some((3, 7)));
        // Min=1+2+3, Max=4+3+2
        assert_eq!(sval_sum_bound::<1, 4>(3), Some((6, 9)));
        // Min=Max=1+2+3+4
        assert_eq!(sval_sum_bound::<1, 4>(4), Some((10, 10)));

        // 1
        assert_eq!(sval_len_bound::<1, 4>(1), Some((1, 1)));
        // 2
        assert_eq!(sval_len_bound::<1, 4>(2), Some((1, 1)));
        // 3, 1+2
        assert_eq!(sval_len_bound::<1, 4>(3), Some((1, 2)));
        // 4, 1+3
        assert_eq!(sval_len_bound::<1, 4>(4), Some((1, 2)));
        // 1+4, 2+3
        assert_eq!(sval_len_bound::<1, 4>(5), Some((2, 2)));
        // 2+4, 1+2+3
        assert_eq!(sval_len_bound::<1, 4>(6), Some((2, 3)));
        // 3+4, 1+2+4
        assert_eq!(sval_len_bound::<1, 4>(7), Some((2, 3)));
        // 1+3+4
        assert_eq!(sval_len_bound::<1, 4>(8), Some((3, 3)));
        // 2+3+4
        assert_eq!(sval_len_bound::<1, 4>(9), Some((3, 3)));
        // 1+2+3+4
        assert_eq!(sval_len_bound::<1, 4>(10), Some((4, 4)));
    }

    #[test]
    fn test_sudoku_grid() {
        let mut sudoku: SState<9, 9, 1, 9> = SState::new();
        assert_eq!(sudoku.apply([0, 0], SVal(5)), Ok(()));
        assert_eq!(sudoku.apply([8, 8], SVal(1)), Ok(()));
        assert_eq!(sudoku.get([0, 0]), Some(SVal(5)));
        assert_eq!(sudoku.undo([0, 0], SVal(5)), Ok(()));
        assert_eq!(sudoku.get([0, 0]), None);
        assert_eq!(sudoku.get([8, 8]), Some(SVal(1)));
        sudoku.reset();
        assert_eq!(sudoku.get([8, 8]), None);
    }

    fn apply<U: UInt, V: Value<U>>(s1: &mut dyn Stateful<U, V>, s2: &mut dyn Stateful<U, V>, index: Index, value: V) {
        s1.apply(index, value).unwrap();
        s2.apply(index, value).unwrap();
    }

    #[test]
    fn test_sudoku_row_violation() {
        let mut sudoku: SState<9, 9, 1, 9> = SState::new();
        let mut checker = RowColChecker::new();
        apply(&mut sudoku, &mut checker, [5, 3], SVal(1));
        apply(&mut sudoku, &mut checker, [5, 4], SVal(3));
        let mut grid = DecisionGrid::new(9, 9);
        assert!(checker.check(&sudoku, &mut grid).is_ok());
        apply(&mut sudoku, &mut checker, [5, 8], SVal(1));
        assert_eq!(checker.check(&sudoku, &mut grid), ConstraintResult::Contradiction);
    }

    #[test]
    fn test_sudoku_col_violation() {
        let mut sudoku: SState<9, 9, 1, 9> = SState::new();
        let mut checker = RowColChecker::new();
        apply(&mut sudoku, &mut checker, [1, 3], SVal(2));
        apply(&mut sudoku, &mut checker, [3, 3], SVal(7));
        let mut grid = DecisionGrid::new(9, 9);
        assert!(checker.check(&sudoku, &mut grid).is_ok());
        apply(&mut sudoku, &mut checker, [6, 3], SVal(2));
        assert_eq!(checker.check(&sudoku, &mut grid), ConstraintResult::Contradiction);
    }

    #[test]
    fn test_sudoku_box_violation() {
        let mut sudoku: SState<9, 9, 1, 9> = SState::new();
        let mut checker = NineBoxChecker::new();
        apply(&mut sudoku, &mut checker, [3, 0], SVal(8));
        apply(&mut sudoku, &mut checker, [4, 1], SVal(2));
        let mut grid = DecisionGrid::new(9, 9);
        assert!(checker.check(&sudoku, &mut grid).is_ok());
        apply(&mut sudoku, &mut checker, [5, 2], SVal(8));
        assert_eq!(checker.check(&sudoku, &mut grid), ConstraintResult::Contradiction);
    }

    #[test]
    fn test_sudoku_parse() {
        let input: &str = "5.3......\n\
                           6..195...\n\
                           .98....6.\n\
                           8...6...3\n\
                           4..8.3..1\n\
                           7...2...6\n\
                           .6....28.\n\
                           ...419..5\n\
                           ......8.9\n";
        let sudoku: SState<9,9, 1,9> = SState::parse(input).unwrap();
        assert_eq!(sudoku.get([0, 0]), Some(SVal::new(5)));
        assert_eq!(sudoku.get([8, 8]), Some(SVal::new(9)));
        assert_eq!(sudoku.get([2, 7]), Some(SVal::new(6)));
        assert_eq!(sudoku.to_string(), input);
    }

    #[test]
    fn test_nine_solve() {
        // #t1d1p1 from sudoku-puzzles.net
        let input: &str = ".7.583.2.\n\
                           .592..3..\n\
                           34...65.7\n\
                           795...632\n\
                           ..36971..\n\
                           68...27..\n\
                           914835.76\n\
                           .3.7.1495\n\
                           567429.13\n";
        let mut sudoku = nine_standard_parse(input).unwrap();
        let ranker = LinearRanker::default();
        let mut checker = nine_standard_checker();
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &ranker, &mut checker, None);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.get_state().get([2, 2]), Some(SVal::new(2)));
                assert_eq!(solved.get_state().get([2, 3]), Some(SVal::new(9)));
                assert_eq!(solved.get_state().get([2, 4]), Some(SVal::new(1)));
            }
            Err(e) => panic!("Failed to solve sudoku: {:?}", e),
        }
    }

    #[test]
    fn test_eight_solve() {
        // #t34d1p1 from sudoku-puzzles.net
        let input: &str = "2...1.38\n\
                           316..7.2\n\
                           .45...8.\n\
                           1..26475\n\
                           ..475...\n\
                           52..7.6.\n\
                           .713...6\n\
                           46..8...\n";
        let mut sudoku = eight_standard_parse(input).unwrap();
        let ranker = LinearRanker::default();
        let mut checker = eight_standard_checker();
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &ranker, &mut checker, None);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.get_state().get([6, 4]), Some(SVal::new(2)));
                assert_eq!(solved.get_state().get([6, 5]), Some(SVal::new(5)));
                assert_eq!(solved.get_state().get([6, 6]), Some(SVal::new(4)));
            }
            Err(e) => panic!("Failed to solve sudoku: {:?}", e),
        }
    }

    #[test]
    fn test_six_solve() {
        // #t2d1p1 from sudoku-puzzles.net
        let input: &str = ".3.4..\n\
                           ..56.3\n\
                           ...1..\n\
                           .1.3.5\n\
                           .64.31\n\
                           ..1.46\n";
        let mut sudoku = six_standard_parse(input).unwrap();
        let ranker = LinearRanker::default();
        let mut checker = six_standard_checker();
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &ranker, &mut checker, None);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.get_state().get([2, 0]), Some(SVal::new(6)));
                assert_eq!(solved.get_state().get([2, 1]), Some(SVal::new(5)));
                assert_eq!(solved.get_state().get([2, 2]), Some(SVal::new(3)));
            }
            Err(e) => panic!("Failed to solve sudoku: {:?}", e),
        }
    }

    #[test]
    fn test_four_solve() {
        // #t14d1p1 from sudoku-puzzles.net
        let input: &str = "...4\n\
                           ....\n\
                           2..3\n\
                           4.12\n";
        let mut sudoku = four_standard_parse(input).unwrap();
        let ranker = LinearRanker::default();
        let mut checker = four_standard_checker();
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &ranker, &mut checker, None);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.get_state().get([0, 0]), Some(SVal::new(1)));
                assert_eq!(solved.get_state().get([0, 1]), Some(SVal::new(2)));
                assert_eq!(solved.get_state().get([0, 2]), Some(SVal::new(3)));
            }
            Err(e) => panic!("Failed to solve sudoku: {:?}", e),
        }
    }
}