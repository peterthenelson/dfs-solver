use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::sync::{LazyLock, Mutex};
use crate::core::{full_set, to_value, unpack_values, Attribution, ConstraintResult, DecisionGrid, Error, Index, State, Stateful, UVGrid, UVSet, UVUnwrapped, UVWrapped, UVal, Value, WithId};
use crate::constraint::Constraint;

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

    fn possibilities() -> Vec<Self> {
        (MIN..=MAX).map(|v| SVal(v)).collect()
    }

    fn from_uval(u: UVal<u8, UVUnwrapped>) -> Self {
        SVal(u.value() + MIN)
    }

    fn to_uval(self) -> UVal<u8, UVWrapped> {
        UVal::new(self.0 - MIN)
    }
}

pub fn unpack_sval_vals<const MIN: u8, const MAX: u8>(s: &UVSet<u8>) -> Vec<u8> {
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

pub trait Overlay: Clone + Debug {
    type Iter<'a>: Iterator<Item = Index> where Self: 'a;
    fn partition_dimension(&self) -> usize;
    fn n_partitions(&self, dim: usize) -> usize;
    fn enclosing_partition(&self, index: Index, dim: usize) -> Option<usize>;
    fn enclosing_partitions(&self, index: Index) -> Vec<Option<usize>> {
        (0..self.partition_dimension())
            .map(|dim| self.enclosing_partition(index, dim))
            .collect()
    }
    fn partition_iter(&self, dim: usize, index: usize) -> Self::Iter<'_>;
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        for dim in 0..self.partition_dimension() {
            if self.enclosing_partition(i1, dim) == self.enclosing_partition(i2, dim) {
                return true;
            }
        }
        false
    }
    fn all_mutually_visible(&self, indices: &Vec<Index>) -> bool {
        indices.iter().all(|i| self.mutually_visible(indices[0], *i))
    }
}

/// Standard rectangular Sudoku grid.
#[derive(Clone)]
pub struct SState<const N: usize, const M: usize, const MIN: u8, const MAX: u8, O: Overlay> {
    grid: UVGrid<u8>,
    overlay: O,
} 

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8, O: Overlay> Debug for SState<N, M, MIN, MAX, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.serialize())
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8, O: Overlay> SState<N, M, MIN, MAX, O> {
    pub fn new(overlay: O) -> Self {
        Self { grid: UVGrid::new(N, M), overlay }
    }

    pub fn get_overlay(&self) -> &O { &self.overlay }

    pub fn parse(s: &str, overlay: O) -> Result<Self, Error> {
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
        Ok(Self { grid, overlay })
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

pub type NineStd = SState<9, 9, 1, 9, StandardSudokuOverlay<9, 9>>;
pub type EightStd = SState<8, 8, 1, 8, StandardSudokuOverlay<8, 8>>;
pub type SixStd = SState<6, 6, 1, 6, StandardSudokuOverlay<6, 6>>;
pub type FourStd = SState<4, 4, 1, 4, StandardSudokuOverlay<4, 4>>;

pub fn nine_standard_parse(s: &str) -> Result<NineStd, Error> {
    NineStd::parse(s, nine_standard_overlay())
}
pub fn eight_standard_parse(s: &str) -> Result<EightStd, Error> {
    EightStd::parse(s, eight_standard_overlay())
}
pub fn six_standard_parse(s: &str) -> Result<SixStd, Error> {
    SixStd::parse(s, six_standard_overlay())
}
pub fn four_standard_parse(s: &str) -> Result<FourStd, Error> {
    FourStd::parse(s, four_standard_overlay())
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8, O: Overlay>
Display for SState<N, M, MIN, MAX, O> {
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

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8, O: Overlay>
State<u8> for SState<N, M, MIN, MAX, O> {
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

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8, O: Overlay>
Stateful<u8, SVal<MIN, MAX>> for SState<N, M, MIN, MAX, O> {
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

#[derive(Clone, Copy, Debug)]
pub struct StandardSudokuOverlay<const N: usize, const M: usize> {
    br: usize,
    bc: usize,
    bh: usize,
    bw: usize,
}

enum StandardSudokuOverlayIteratorState {
    Row(usize, Option<Index>, usize),
    Col(usize, Option<Index>, usize),
    Box(usize, Option<Index>, usize, usize),
}

pub struct StandardSudokuOverlayIterator<'a, const N: usize, const M: usize> {
    overlay: &'a StandardSudokuOverlay<N, M>,
    state: StandardSudokuOverlayIteratorState,
}

impl <'a, const N: usize, const M: usize> StandardSudokuOverlayIterator<'a, N, M> {
    pub fn row(overlay: &'a StandardSudokuOverlay<N, M>, r: usize) -> Self {
        Self {
            overlay,
            state: StandardSudokuOverlayIteratorState::Row(r, None, 0),
        }
    }

    pub fn others_in_row(overlay: &'a StandardSudokuOverlay<N, M>, cell: Index) -> Self {
        Self {
            overlay,
            state: StandardSudokuOverlayIteratorState::Row(cell[0], Some(cell), 0),
        }
    }

    pub fn col(overlay: &'a StandardSudokuOverlay<N, M>, c: usize) -> Self {
        Self {
            overlay,
            state: StandardSudokuOverlayIteratorState::Col(c, None, 0),
        }
    }

    pub fn others_in_col(overlay: &'a StandardSudokuOverlay<N, M>, cell: Index) -> Self {
        Self {
            overlay,
            state: StandardSudokuOverlayIteratorState::Col(cell[1], Some(cell), 0),
        }
    }

    pub fn box_(overlay: &'a StandardSudokuOverlay<N, M>, b: usize) -> Self {
        Self {
            overlay,
            state: StandardSudokuOverlayIteratorState::Box(b, None, 0, 0),
        }
    }

    pub fn others_in_box(overlay: &'a StandardSudokuOverlay<N, M>, cell: Index) -> Self {
        let (b, box_index) = overlay.to_box_coords(cell);
        Self {
            overlay,
            state: StandardSudokuOverlayIteratorState::Box(b, Some(box_index), 0, 0),
        }
    }
}

impl <'a, const N: usize, const M: usize> Iterator for StandardSudokuOverlayIterator<'a, N, M> {
    type Item = Index;
    fn next(&mut self) -> Option<Self::Item> {
        let ret: Index;
        match self.state {
            StandardSudokuOverlayIteratorState::Row(r, skip, c) => {
                if c >= self.overlay.cols() {
                    return None;
                }
                if let Some(skip_index) = skip {
                    if skip_index[1] == c {
                        self.state = StandardSudokuOverlayIteratorState::Row(r, skip, c+1);
                        return self.next();
                    }
                }
                ret = [r, c];
                self.state = StandardSudokuOverlayIteratorState::Row(r, skip, c+1);
            },
            StandardSudokuOverlayIteratorState::Col(c, skip, r) => {
                if r >= self.overlay.rows() {
                    return None;
                }
                if let Some(skip_index) = skip {
                    if skip_index[0] == r {
                        self.state = StandardSudokuOverlayIteratorState::Col(c, skip, r+1);
                        return self.next();
                    }
                }
                ret = [r, c];
                self.state = StandardSudokuOverlayIteratorState::Col(c, skip, r+1);
            },
            StandardSudokuOverlayIteratorState::Box(b, skip, br, bc) => {
                let (bh, bw) = self.overlay.box_dims();
                if br >= bh {
                    return None
                }
                if let Some(skip_index) = skip {
                    if skip_index[0] == br && skip_index[1] == bc {
                        self.state = if bc + 1 == bw {
                            StandardSudokuOverlayIteratorState::Box(b, skip, br+1, 0)
                        } else {
                            StandardSudokuOverlayIteratorState::Box(b, skip, br, bc+1)
                        };
                        return self.next();
                    }
                }
                ret = self.overlay.from_box_coords(b, [br, bc]);
                self.state = if bc + 1 == bw {
                    StandardSudokuOverlayIteratorState::Box(b, skip, br+1, 0)
                } else {
                    StandardSudokuOverlayIteratorState::Box(b, skip, br, bc+1)
                };
            },
        }
        Some(ret)
    }
}

impl <const N: usize, const M: usize> StandardSudokuOverlay<N, M> {
    pub fn new(br: usize, bc: usize, bh: usize, bw: usize) -> Self {
        if N != br * bh {
            panic!("StandardSudokuOverlay expected N == br*bh, but {} != {}*{}", N, br, bh);
        } else if M != bc * bw {
            panic!("StandardSudokuOverlay expected M == bc*bw, but {} != {}*{}", M, bc, bw);
        }
        Self { br, bc, bh, bw }
    }
    pub const fn rows(&self) -> usize { N }
    pub fn row_iter(&self, r: usize) -> StandardSudokuOverlayIterator<N, M> {
        StandardSudokuOverlayIterator::row(self, r)
    }
    pub fn others_in_row(&self, cell: Index) -> StandardSudokuOverlayIterator<N, M> {
        StandardSudokuOverlayIterator::others_in_row(self, cell)
    }
    pub const fn cols(&self) -> usize { M }
    pub fn col_iter(&self, c: usize) -> StandardSudokuOverlayIterator<N, M> {
        StandardSudokuOverlayIterator::col(self, c)
    }
    pub fn others_in_col(&self, cell: Index) -> StandardSudokuOverlayIterator<N, M> {
        StandardSudokuOverlayIterator::others_in_col(self, cell)
    }
    pub fn boxes(&self) -> usize { self.br * self.bc }
    pub fn box_dims(&self) -> (usize, usize) {
        (self.bh, self.bw)
    }
    /// Get which box an index is in, along with the coordinates within that box.
    pub fn to_box_coords(&self, index: Index) -> (usize, Index) {
        (self.bc*(index[0] / self.bh) + (index[1] / self.bw), [index[0] % self.bh, index[1] % self.bw])
    }
    /// Given a box and coordinates within it, get the index in the grid.
    pub fn from_box_coords(&self, box_index: usize, index: Index) -> Index {
        let r = (box_index / self.bc) * self.bh + index[0];
        let c = (box_index % self.bc) * self.bw + index[1];
        [r, c]
    }
    pub fn box_iter(&self, b: usize) -> StandardSudokuOverlayIterator<N, M> {
        StandardSudokuOverlayIterator::box_(self, b)
    }
    pub fn others_in_box(&self, cell: Index) -> StandardSudokuOverlayIterator<N, M> {
        StandardSudokuOverlayIterator::others_in_box(self, cell)
    }
}

impl <const N: usize, const M: usize> Overlay for StandardSudokuOverlay<N, M> {
    type Iter<'a> = StandardSudokuOverlayIterator<'a, N, M> where Self: 'a;
    /// There are rows, columns, and boxes.
    fn partition_dimension(&self) -> usize { 3 }
    fn n_partitions(&self, dim: usize) -> usize {
        match dim {
            0 => self.rows(),
            1 => self.cols(),
            2 => self.boxes(),
            _ => panic!("Invalid dimension for StandardSudokuOverlay: {}", dim),
        }
    }
    fn enclosing_partition(&self, index: Index, dim: usize) -> Option<usize> {
        match dim {
            0 => Some(index[0]),
            1 => Some(index[1]),
            2 => {
                let (b, _) = self.to_box_coords(index);
                Some(b)
            },
            _ => panic!("Invalid dimension for StandardSudokuOverlay: {}", dim),
        }
    }
    fn partition_iter(&self, dim: usize, index: usize) -> Self::Iter<'_> {
        match dim {
            0 => self.row_iter(index),
            1 => self.col_iter(index),
            2 => self.box_iter(index),
            _ => panic!("Invalid dimension for StandardSudokuOverlay: {}", dim),
        }
    }
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        if i1[0] == i2[0] || i1[1] == i2[1] {
            return true;
        }
        let (b1, _) = self.to_box_coords(i1);
        let (b2, _) = self.to_box_coords(i2);
        b1 == b2
    }
}

pub fn nine_standard_overlay() -> StandardSudokuOverlay<9, 9> {
    StandardSudokuOverlay::new(3, 3, 3, 3)
}
pub fn eight_standard_overlay() -> StandardSudokuOverlay<8, 8> {
    StandardSudokuOverlay::new(4, 2, 2, 4)
}
pub fn six_standard_overlay() -> StandardSudokuOverlay<6, 6> {
    StandardSudokuOverlay::new(3, 2, 2, 3)
}
pub fn four_standard_overlay() -> StandardSudokuOverlay<4, 4> {
    StandardSudokuOverlay::new(2, 2, 2, 2)
}

pub const ROW_CONFLICT_ATTRIBUTION: &str = "ROW_CONFLICT";
pub const COL_CONFLICT_ATTRIBUTION: &str = "COL_CONFLICT";
pub const BOX_CONFLICT_ATTRIBUTION: &str = "BOX_CONFLICT";

pub struct StandardSudokuChecker<const N: usize, const M: usize, const MIN: u8, const MAX: u8> {
    overlay: StandardSudokuOverlay<N, M>,
    row: [UVSet<u8>; N],
    col: [UVSet<u8>; M],
    boxes: Box<[UVSet<u8>]>,
    row_attribution: Attribution<WithId>,
    col_attribution: Attribution<WithId>,
    box_attribution: Attribution<WithId>,
    illegal: Option<(Index, SVal<MIN, MAX>, Attribution<WithId>)>,
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> StandardSudokuChecker<N, M, MIN, MAX> {
    pub fn new(state: &SState<N, M, MIN, MAX, StandardSudokuOverlay<N, M>>) -> Self {
        return Self {
            overlay: state.get_overlay().clone(),
            row: std::array::from_fn(|_| full_set::<u8, SVal<MIN, MAX>>()),
            col: std::array::from_fn(|_| full_set::<u8, SVal<MIN, MAX>>()),
            boxes: vec![full_set::<u8, SVal<MIN, MAX>>(); state.get_overlay().boxes()].into_boxed_slice(),
            row_attribution: Attribution::new(ROW_CONFLICT_ATTRIBUTION).unwrap(),
            col_attribution: Attribution::new(COL_CONFLICT_ATTRIBUTION).unwrap(),
            box_attribution: Attribution::new(BOX_CONFLICT_ATTRIBUTION).unwrap(),
            illegal: None,
        }
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8>
Debug for StandardSudokuChecker<N, M, MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((i, v, a)) = &self.illegal {
            write!(f, "Illegal move: {:?}={:?} ({})\n", i, v, a.get_name())?;
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
        write!(f, "Unused vals by box:\n")?;
        for b in 0..self.overlay.boxes() {
            let vals = unpack_sval_vals::<MIN, MAX>(&self.boxes[b]);
            write!(f, " {}: {:?}\n", b, vals)?;
        }
        Ok(())
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8>
Stateful<u8, SVal<MIN, MAX>> for StandardSudokuChecker<N, M, MIN, MAX> {
    fn reset(&mut self) {
        self.row = std::array::from_fn(|_| full_set::<u8, SVal<MIN, MAX>>());
        self.col = std::array::from_fn(|_| full_set::<u8, SVal<MIN, MAX>>());
        self.boxes = vec![full_set::<u8, SVal<MIN, MAX>>(); self.overlay.boxes()].into_boxed_slice();
        self.illegal = None;
    }

    fn apply(&mut self, index: Index, value: SVal<MIN, MAX>) -> Result<(), Error> {
        let uv = value.to_uval();
        // In theory we could be allow multiple illegal moves and just
        // invalidate and recalculate the grid or something, but it seems hard.
        if self.illegal.is_some() {
            return Err(ILLEGAL_ACTION_RC);
        }
        let (b, _) = self.overlay.to_box_coords(index);
        if !self.row[index[0]].contains(uv) {
            self.illegal = Some((index, value, self.row_attribution.clone()));
            return Ok(());
        } else if !self.col[index[1]].contains(uv) {
            self.illegal = Some((index, value, self.col_attribution.clone()));
            return Ok(());
        } else if !self.boxes[b].contains(uv){
            self.illegal = Some((index, value, self.box_attribution.clone()));
            return Ok(());
        }
        self.row[index[0]].remove(uv);
        self.col[index[1]].remove(uv);
        self.boxes[b].remove(uv);
        Ok(())
    }

    fn undo(&mut self, index: Index, value: SVal<MIN, MAX>) -> Result<(), Error> {
        if let Some((i, v, _)) = self.illegal {
            if i != index || v != value {
                return Err(UNDO_MISMATCH);
            } else {
                self.illegal = None;
                return Ok(());
            }
        }
        let uv = value.to_uval();
        let (b, _) = self.overlay.to_box_coords(index);
        self.row[index[0]].insert(uv);
        self.col[index[1]].insert(uv);
        self.boxes[b].insert(uv);
        Ok(())
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8, O: Overlay>
Constraint<u8, SState<N, M, MIN, MAX, O>> for StandardSudokuChecker<N, M, MIN, MAX> {
    fn check(&self, puzzle: &SState<N, M, MIN, MAX, O>, grid: &mut DecisionGrid<u8, SVal<MIN, MAX>>) -> ConstraintResult<u8, SVal<MIN, MAX>> {
        if let Some((_, _, a)) = &self.illegal {
            return ConstraintResult::Contradiction(a.clone());
        }
        for r in 0..N {
            for c in 0..M {
                if puzzle.get([r, c]).is_some() {
                    continue;
                }
                let cell = grid.get_mut([r, c]);
                let (b, _) = self.overlay.to_box_coords([r, c]);
                cell.0.intersect_with(&self.row[r]);
                cell.0.intersect_with(&self.col[c]);
                cell.0.intersect_with(&self.boxes[b]);
            }
        }
        ConstraintResult::Ok
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{core::{empty_set, UInt}, ranker::LinearRanker, solver::FindFirstSolution};
    use crate::core::test_util::round_trip_value;
    use crate::constraint::test_util::assert_contradiction;

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
        let mut sudoku: SState<9, 9, 1, 9, _> = SState::new(nine_standard_overlay());
        assert_eq!(sudoku.apply([0, 0], SVal(5)), Ok(()));
        assert_eq!(sudoku.apply([8, 8], SVal(1)), Ok(()));
        assert_eq!(sudoku.get([0, 0]), Some(SVal(5)));
        assert_eq!(sudoku.undo([0, 0], SVal(5)), Ok(()));
        assert_eq!(sudoku.get([0, 0]), None);
        assert_eq!(sudoku.get([8, 8]), Some(SVal(1)));
        sudoku.reset();
        assert_eq!(sudoku.get([8, 8]), None);
    }

    #[test]
    fn test_sudoku_overlay() {
        let overlay = StandardSudokuOverlay::<9, 9>::new(3, 3, 3, 3);
        assert_eq!(overlay.rows(), 9);
        assert_eq!(overlay.cols(), 9);
        assert_eq!(overlay.boxes(), 9);
        assert_eq!(overlay.box_dims(), (3, 3));
        assert_eq!(overlay.to_box_coords([7, 4]), (7, [1, 1]));
        assert_eq!(overlay.from_box_coords(7, [1, 1]), [7, 4]);
        assert_eq!(
            overlay.row_iter(2).collect::<Vec<_>>(),
            vec![[2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8]],
        );
        assert_eq!(
            overlay.others_in_row([2, 3]).collect::<Vec<_>>(),
            vec![[2, 0], [2, 1], [2, 2], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8]],
        );
        assert_eq!(
            overlay.col_iter(2).collect::<Vec<_>>(),
            vec![[0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2]],
        );
        assert_eq!(
            overlay.others_in_col([3, 2]).collect::<Vec<_>>(),
            vec![[0, 2], [1, 2], [2, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2]],
        );
        assert_eq!(
            overlay.box_iter(7).collect::<Vec<_>>(),
            vec![[6, 3], [6, 4], [6, 5], [7, 3], [7, 4], [7, 5], [8, 3], [8, 4], [8, 5]],
        );
        assert_eq!(
            overlay.others_in_box([7, 4]).collect::<Vec<_>>(),
            vec![[6, 3], [6, 4], [6, 5], [7, 3], [7, 5], [8, 3], [8, 4], [8, 5]],
        );
        assert!(overlay.mutually_visible([7, 4], [7, 8]));
        assert!(overlay.mutually_visible([7, 4], [1, 4]));
        assert!(overlay.mutually_visible([7, 4], [6, 3]));
    }

    fn apply2<U: UInt, V: Value<U>>(s1: &mut dyn Stateful<U, V>, s2: &mut dyn Stateful<U, V>, index: Index, value: V) {
        s1.apply(index, value).unwrap();
        s2.apply(index, value).unwrap();
    }

    #[test]
    fn test_sudoku_row_violation() {
        let mut sudoku: SState<9, 9, 1, 9, _> = SState::new(nine_standard_overlay());
        let mut checker = StandardSudokuChecker::new(&sudoku);
        apply2(&mut sudoku, &mut checker, [5, 3], SVal(1));
        apply2(&mut sudoku, &mut checker, [5, 4], SVal(3));
        let mut grid = DecisionGrid::new(9, 9);
        assert!(checker.check(&sudoku, &mut grid).is_ok());
        apply2(&mut sudoku, &mut checker, [5, 8], SVal(1));
        assert_contradiction(checker.check(&sudoku, &mut grid), "ROW_CONFLICT");
    }

    #[test]
    fn test_sudoku_col_violation() {
        let mut sudoku: SState<9, 9, 1, 9, _> = SState::new(nine_standard_overlay());
        let mut checker = StandardSudokuChecker::new(&sudoku);
        apply2(&mut sudoku, &mut checker, [1, 3], SVal(2));
        apply2(&mut sudoku, &mut checker, [3, 3], SVal(7));
        let mut grid = DecisionGrid::new(9, 9);
        assert!(checker.check(&sudoku, &mut grid).is_ok());
        apply2(&mut sudoku, &mut checker, [6, 3], SVal(2));
        assert_contradiction(checker.check(&sudoku, &mut grid), "COL_CONFLICT");
    }

    #[test]
    fn test_sudoku_box_violation() {
        let mut sudoku: SState<9, 9, 1, 9, _> = SState::new(nine_standard_overlay());
        let mut checker = StandardSudokuChecker::new(&sudoku);
        apply2(&mut sudoku, &mut checker, [3, 0], SVal(8));
        apply2(&mut sudoku, &mut checker, [4, 1], SVal(2));
        let mut grid = DecisionGrid::new(9, 9);
        assert!(checker.check(&sudoku, &mut grid).is_ok());
        apply2(&mut sudoku, &mut checker, [5, 2], SVal(8));
        assert_contradiction(checker.check(&sudoku, &mut grid), "BOX_CONFLICT");
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
        let sudoku = nine_standard_parse(input).unwrap();
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
        let mut checker = StandardSudokuChecker::new(&sudoku);
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
        let mut checker = StandardSudokuChecker::new(&sudoku);
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
        let mut checker = StandardSudokuChecker::new(&sudoku);
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
        let mut checker = StandardSudokuChecker::new(&sudoku);
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