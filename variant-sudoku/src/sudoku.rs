use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::sync::{LazyLock, Mutex};

use crate::core::{Attribution, ConstraintResult, Error, Index, Key, Overlay, RegionLayer, State, Stateful, UVUnwrapped, UVWrapped, UVal, VBitSet, VSet, VSetMut, Value, BOXES_LAYER, COLS_LAYER, ROWS_LAYER};
use crate::constraint::Constraint;
use crate::index_util::parse_val_grid;
use crate::ranker::RankingInfo;

impl <const MIN: u8, const MAX: u8> Display for StdVal<MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.val())
    }
}

/// Standard Sudoku value, ranging from a minimum to a maximum value (inclusive).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct StdVal<const MIN: u8, const MAX: u8>(u8);

impl <const MIN: u8, const MAX: u8> StdVal<MIN, MAX> {
    pub fn new(value: u8) -> Self {
        assert!(value >= MIN && value <= MAX, "Value out of bounds");
        StdVal(value)
    }

    pub fn val(self) -> u8 {
        self.0
    }
}

impl <const MIN: u8, const MAX: u8> Value for StdVal<MIN, MAX> {
    type U = u8;

    fn parse(s: &str) -> Result<Self, Error> {
        let value = s.parse::<u8>().map_err(|v| Error::new(format!("Invalid value: {}", v).to_string()))?;
        if value < MIN || value > MAX {
            return Err(Error::new(format!("Value out of bounds: {} ({}-{})", value, MIN, MAX)));
        }
        Ok(StdVal(value))
    }

    fn cardinality() -> usize {
        (MAX - MIN + 1) as usize
    }

    fn nth(ord: usize) -> Self {
        Self::new(ord as u8 + MIN)
    }

    fn possibilities() -> Vec<Self> {
        (MIN..=MAX).map(|v| StdVal(v)).collect()
    }

    fn ordinal(&self) -> usize {
        (self.val() - MIN) as usize
    }

    fn from_uval(u: UVal<u8, UVUnwrapped>) -> Self {
        StdVal(u.value() + MIN)
    }

    fn to_uval(self) -> UVal<u8, UVWrapped> {
        UVal::new(self.0 - MIN)
    }
}

pub fn unpack_stdval_vals<const MIN: u8, const MAX: u8, VS: VSet<StdVal<MIN, MAX>>>(s: &VS) -> Vec<u8> {
    s.iter().map(|v| v.val()).collect::<Vec<u8>>()
}

/// Tables of useful sums in Sudoku.
static STDVAL_LEN_TO_SUM: LazyLock<Mutex<HashMap<(u8, u8), HashMap<u8, Option<(u8, u8)>>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});
static STDVAL_SUM_TO_LEN: LazyLock<Mutex<HashMap<(u8, u8), HashMap<u8, Option<(u8, u8)>>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

// Useful utility for working with sums of ranges of numbers. Returns the range
// of sums that are possible to from len exclusive numbers drawn from the range
// min..=max.
pub fn min_max_sum(min: u8, max: u8, len: u8) -> Option<(u8, u8)> {
    if min + len - 1 > max {
        None
    } else {
        Some((
            ((min+len-1)*(min+len)-min*(min-1))/2,
            (max*(max+1)-(max-len)*(max+1-len))/2,
        ))
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

// What is the minimum/maximum sum that len exclusive StdVal<MIN, MAX>s could add up to?
pub fn stdval_sum_bound<const MIN: u8, const MAX: u8>(len: u8) -> Option<(u8, u8)> {
    let mut map = STDVAL_LEN_TO_SUM.lock().unwrap();
    let inner_map = map.entry((MIN, MAX)).or_default();
    if let Some(r) = inner_map.get(&len) {
        return *r
    }
    let r = min_max_sum(MIN, MAX, len);
    inner_map.insert(len, r);
    r
}

// What is the minimum/maximum len of exclusive StdVal<MIN, MAX>s that could add up to sum?
pub fn stdval_len_bound<const MIN: u8, const MAX: u8>(sum: u8) -> Option<(u8, u8)> {
    let mut map = STDVAL_SUM_TO_LEN.lock().unwrap();
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

pub type NineStdVal = StdVal<1, 9>;
pub type NineStdOverlay = StdOverlay<9, 9>;
pub type NineStd = State<NineStdVal, NineStdOverlay>;
pub type EightStdVal = StdVal<1, 8>;
pub type EightStdOverlay = StdOverlay<8, 8>;
pub type EightStd = State<EightStdVal, EightStdOverlay>;
pub type SixStdVal = StdVal<1, 6>;
pub type SixStdOverlay = StdOverlay<6, 6>;
pub type SixStd = State<SixStdVal, SixStdOverlay>;
pub type FourStdVal = StdVal<1, 4>;
pub type FourStdOverlay = StdOverlay<4, 4>;
pub type FourStd = State<FourStdVal, FourStdOverlay>;

pub fn nine_standard_parse(s: &str) -> Result<NineStd, Error> {
    nine_standard_overlay().parse_state(s)
}
pub fn eight_standard_parse(s: &str) -> Result<EightStd, Error> {
    eight_standard_overlay().parse_state(s)
}
pub fn six_standard_parse(s: &str) -> Result<SixStd, Error> {
    six_standard_overlay().parse_state(s)
}
pub fn four_standard_parse(s: &str) -> Result<FourStd, Error> {
    four_standard_overlay().parse_state(s)
}
pub fn nine_standard_empty() -> NineStd {
    NineStd::new(nine_standard_overlay())
}
pub fn eight_standard_empty() -> EightStd {
    EightStd::new(eight_standard_overlay())
}
pub fn six_standard_empty() -> SixStd {
    SixStd::new(six_standard_overlay())
}
pub fn four_standard_empty() -> FourStd {
    FourStd::new(four_standard_overlay())
}

pub const UNDO_MISMATCH_ERROR: Error = Error::new_const("Undo value mismatch");
pub const ILLEGAL_ACTION_RC: Error = Error::new_const("A row/col violation already exists; can't apply further actions.");
pub const ILLEGAL_ACTION_BOX: Error = Error::new_const("A box violation already exists; can't apply further actions.");

#[derive(Clone, Debug)]
pub struct StdOverlay<const N: usize, const M: usize> {
    br: usize,
    bc: usize,
    bh: usize,
    bw: usize,
    custom_layers: Vec<(Key<RegionLayer>, Vec<(bool, Vec<Index>)>)>,
}

enum StdOverlayIteratorState {
    Row(usize, Option<Index>, usize),
    Col(usize, Option<Index>, usize),
    Box(usize, Option<Index>, usize, usize),
    Custom(usize, usize, Option<Index>, usize),
}

pub struct StdOverlayIterator<'a, const N: usize, const M: usize> {
    overlay: &'a StdOverlay<N, M>,
    state: StdOverlayIteratorState,
}

impl <'a, const N: usize, const M: usize> StdOverlayIterator<'a, N, M> {
    pub fn row(overlay: &'a StdOverlay<N, M>, r: usize) -> Self {
        Self {
            overlay,
            state: StdOverlayIteratorState::Row(r, None, 0),
        }
    }

    pub fn others_in_row(overlay: &'a StdOverlay<N, M>, cell: Index) -> Self {
        Self {
            overlay,
            state: StdOverlayIteratorState::Row(cell[0], Some(cell), 0),
        }
    }

    pub fn col(overlay: &'a StdOverlay<N, M>, c: usize) -> Self {
        Self {
            overlay,
            state: StdOverlayIteratorState::Col(c, None, 0),
        }
    }

    pub fn others_in_col(overlay: &'a StdOverlay<N, M>, cell: Index) -> Self {
        Self {
            overlay,
            state: StdOverlayIteratorState::Col(cell[1], Some(cell), 0),
        }
    }

    pub fn box_(overlay: &'a StdOverlay<N, M>, b: usize) -> Self {
        Self {
            overlay,
            state: StdOverlayIteratorState::Box(b, None, 0, 0),
        }
    }

    pub fn others_in_box(overlay: &'a StdOverlay<N, M>, cell: Index) -> Self {
        let (b, box_index) = overlay.to_box_coords(cell);
        Self {
            overlay,
            state: StdOverlayIteratorState::Box(b, Some(box_index), 0, 0),
        }
    }

    pub fn custom_region(overlay: &'a StdOverlay<N, M>, layer: Key<RegionLayer>, index: usize) -> Self {
        for (layer_index, cl) in overlay.custom_layers.iter().enumerate() {
            if cl.0 == layer {
                if index >= cl.1.len() {
                    panic!("There are only {} regions in layer {}; got index {}", cl.1.len(), layer.name(), index);
                }
                return Self {
                    overlay,
                    state: StdOverlayIteratorState::Custom(layer_index, index, None, 0),
                }
            }
        }
        panic!("No such layer exists: {}", layer.name());
    }

    pub fn others_in_custom_region(overlay: &'a StdOverlay<N, M>, layer: Key<RegionLayer>, cell: Index) -> Self {
        let (index, _) = overlay.enclosing_region_and_offset(layer, cell).expect(
            format!("Expected cell {:?} to be in a region in layer {}", cell, layer.name()).as_str(),
        );
        for (layer_index, cl) in overlay.custom_layers.iter().enumerate() {
            if cl.0 == layer {
                return Self {
                    overlay,
                    state: StdOverlayIteratorState::Custom(layer_index, index, Some(cell), 0),
                }
            }
        }
        panic!("No such layer exists: {}", layer.name());
    }
}

impl <'a, const N: usize, const M: usize> Iterator for StdOverlayIterator<'a, N, M> {
    type Item = Index;
    fn next(&mut self) -> Option<Self::Item> {
        let ret: Index;
        match self.state {
            StdOverlayIteratorState::Row(r, skip, c) => {
                if c >= self.overlay.cols() {
                    return None;
                }
                if let Some(skip_index) = skip {
                    if skip_index[1] == c {
                        self.state = StdOverlayIteratorState::Row(r, skip, c+1);
                        return self.next();
                    }
                }
                ret = [r, c];
                self.state = StdOverlayIteratorState::Row(r, skip, c+1);
            },
            StdOverlayIteratorState::Col(c, skip, r) => {
                if r >= self.overlay.rows() {
                    return None;
                }
                if let Some(skip_index) = skip {
                    if skip_index[0] == r {
                        self.state = StdOverlayIteratorState::Col(c, skip, r+1);
                        return self.next();
                    }
                }
                ret = [r, c];
                self.state = StdOverlayIteratorState::Col(c, skip, r+1);
            },
            StdOverlayIteratorState::Box(b, skip, br, bc) => {
                let (bh, bw) = self.overlay.box_dims();
                if br >= bh {
                    return None
                }
                if let Some(skip_index) = skip {
                    if skip_index[0] == br && skip_index[1] == bc {
                        self.state = if bc + 1 == bw {
                            StdOverlayIteratorState::Box(b, skip, br+1, 0)
                        } else {
                            StdOverlayIteratorState::Box(b, skip, br, bc+1)
                        };
                        return self.next();
                    }
                }
                ret = self.overlay.from_box_coords(b, [br, bc]);
                self.state = if bc + 1 == bw {
                    StdOverlayIteratorState::Box(b, skip, br+1, 0)
                } else {
                    StdOverlayIteratorState::Box(b, skip, br, bc+1)
                };
            },
            StdOverlayIteratorState::Custom(layer_index, index, skip, i) => {
                let cells = &self.overlay.custom_layers[layer_index].1[index].1;
                if i >= cells.len() {
                    return None;
                }
                if let Some(skip_index) = skip {
                    if skip_index == cells[i] {
                        self.state = StdOverlayIteratorState::Custom(layer_index, index, skip, i+1);
                        return self.next();
                    }
                }
                ret = cells[i];
                self.state = StdOverlayIteratorState::Custom(layer_index, index, skip, i+1);

            },
        }
        Some(ret)
    }
}

impl <const N: usize, const M: usize> StdOverlay<N, M> {
    pub fn new(br: usize, bc: usize, bh: usize, bw: usize) -> Self {
        if N != br * bh {
            panic!("StdOverlay expected N == br*bh, but {} != {}*{}", N, br, bh);
        } else if M != bc * bw {
            panic!("StdOverlay expected M == bc*bw, but {} != {}*{}", M, bc, bw);
        }
        Self { br, bc, bh, bw, custom_layers: Vec::new() }
    }
    pub fn box_rows(&self) -> usize { self.br }
    pub fn box_height(&self) -> usize { self.bh }
    pub fn box_cols(&self) -> usize { self.bc }
    pub fn box_width(&self) -> usize { self.bw }
    pub const fn rows(&self) -> usize { N }
    pub fn row_iter(&self, r: usize) -> StdOverlayIterator<N, M> {
        StdOverlayIterator::row(self, r)
    }
    pub fn others_in_row(&self, cell: Index) -> StdOverlayIterator<N, M> {
        StdOverlayIterator::others_in_row(self, cell)
    }
    pub const fn cols(&self) -> usize { M }
    pub fn col_iter(&self, c: usize) -> StdOverlayIterator<N, M> {
        StdOverlayIterator::col(self, c)
    }
    pub fn others_in_col(&self, cell: Index) -> StdOverlayIterator<N, M> {
        StdOverlayIterator::others_in_col(self, cell)
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
    pub fn box_iter(&self, b: usize) -> StdOverlayIterator<N, M> {
        StdOverlayIterator::box_(self, b)
    }
    pub fn others_in_box(&self, cell: Index) -> StdOverlayIterator<N, M> {
        StdOverlayIterator::others_in_box(self, cell)
    }
    pub fn custom_region_iter(&self, layer: Key<RegionLayer>, index: usize) -> StdOverlayIterator<N, M> {
        StdOverlayIterator::custom_region(self, layer, index)
    }
    pub fn others_in_custom_region(&self, layer: Key<RegionLayer>, cell: Index) -> StdOverlayIterator<N, M> {
        StdOverlayIterator::others_in_custom_region(self, layer, cell)
    }
    pub fn serialize_pretty<V: Value>(&self, s: &State<V, Self>) -> String {
        let mut result = String::new();
        for r in 0..N {
            for c in 0..M {
                if let Some(v) = s.get([r, c]) {
                    result.push_str(v.to_string().as_str())
                } else {
                    result.push('.');
                }
                if c+1 < M {
                    result.push(if (c+1) % self.bw == 0 { '|' } else { ' ' });
                }
            }
            result.push('\n');
            if (r+1) < N && (r+1) % self.bh == 0 {
                for c in 0..M {
                    result.push('-');
                    if c+1 < M {
                        result.push(if (c+1) % self.bw == 0 { '+' } else { '-' });
                    }
                }
                result.push('\n');
            }
        }
        result
    }
}

impl <const N: usize, const M: usize> Overlay for StdOverlay<N, M> {
    type Iter<'a> = StdOverlayIterator<'a, N, M> where Self: 'a;

    fn grid_dims(&self) -> (usize, usize) { (N, M) }

    fn region_layers(&self) -> Vec<Key<RegionLayer>> {
        let mut layers = vec![ROWS_LAYER, COLS_LAYER, BOXES_LAYER];
        if self.custom_layers.is_empty() {
            return layers;
        }
        layers.extend(self.custom_layers.iter().map(|l| l.0));
        layers
    }

    fn add_region_layer(&mut self, layer: Key<RegionLayer>) {
        for cl in &self.custom_layers {
            if cl.0 == layer {
                return;
            }
        }
        self.custom_layers.push((layer, Vec::new()));
    }

    fn regions_in_layer(&self, layer: Key<RegionLayer>) -> usize {
        if layer == ROWS_LAYER {
            self.rows()
        } else if layer == COLS_LAYER {
            self.cols()
        } else if layer == BOXES_LAYER {
            self.boxes()
        } else {
            for cl in &self.custom_layers {
                if cl.0 == layer {
                    return cl.1.len();
                }
            }
            panic!("Invalid region layer for StdOverlay: {}", layer.name())
        }
    }

    fn add_region_in_layer(&mut self, layer: Key<RegionLayer>, positive_constraint: bool, cells: Vec<Index>) -> usize {
        for cl in self.custom_layers.iter_mut() {
            if cl.0 == layer {
                cl.1.push((positive_constraint, cells));
                return cl.1.len() - 1;
            }
        }
        panic!("Invalid region layer for StdOverlay: {}", layer.name())
    }

    fn cells_in_region(&self, layer: Key<RegionLayer>, index: usize) -> usize {
        if layer == ROWS_LAYER {
            self.cols()
        } else if layer == COLS_LAYER {
            self.rows()
        } else if layer == BOXES_LAYER {
            self.bh*self.bw
        } else {
            for cl in &self.custom_layers {
                if cl.0 == layer {
                    return cl.1[index].1.len();
                }
            }
            panic!("Invalid region layer for StdOverlay: {}", layer.name())
        }
    }

    fn has_positive_constraint(&self, layer: Key<RegionLayer>, index: usize) -> bool {
        if layer == ROWS_LAYER {
            true
        } else if layer == COLS_LAYER {
            true
        } else if layer == BOXES_LAYER {
            true
        } else {
            for cl in &self.custom_layers {
                if cl.0 == layer {
                    return cl.1[index].0;
                }
            }
            panic!("Invalid region layer for StdOverlay: {}", layer.name())
        }
        
    }

    fn enclosing_region_and_offset(&self, layer: Key<RegionLayer>, index: Index) -> Option<(usize, usize)> {
        if layer == ROWS_LAYER {
            Some((index[0], index[1]))
        } else if layer == COLS_LAYER {
            Some((index[1], index[0]))
        } else if layer == BOXES_LAYER {
            let (b, [br, bc]) = self.to_box_coords(index);
            Some((b, br*self.bw+bc))
        } else {
            for cl in &self.custom_layers {
                if cl.0 == layer {
                    for (region_index, region) in cl.1.iter().enumerate() {
                        for (offset, cell) in region.1.iter().enumerate() {
                            if *cell == index {
                                return Some((region_index, offset))
                            }
                        }
                    }
                    return None;
                }
            }
            panic!("Invalid region layer for StdOverlay: {}", layer.name())
        }
    }

    fn nth_in_region(&self, layer: Key<RegionLayer>, index: usize, offset: usize) -> Option<Index> {
        if layer == ROWS_LAYER {
            if index < self.rows() && offset < self.cols() {
                Some([index, offset])
            } else {
                None
            }
        } else if layer == COLS_LAYER {
            if index < self.cols() && offset < self.rows() {
                Some([offset, index])
            } else {
                None
            }
        } else if layer == BOXES_LAYER {
            if index < self.boxes() && offset < self.bh*self.bw {
                Some(self.from_box_coords(index, [offset / self.bw, offset % self.bw]))
            } else {
                None
            }
        } else {
            for cl in &self.custom_layers {
                if cl.0 == layer {
                    return Some(cl.1[index].1[offset])
                }
            }
            panic!("Invalid region layer for StdOverlay: {}", layer.name())
        }
    }

    fn region_iter(&self, layer: Key<RegionLayer>, index: usize) -> Self::Iter<'_> {
        if layer == ROWS_LAYER {
            self.row_iter(index)
        } else if layer == COLS_LAYER {
            self.col_iter(index)
        } else if layer == BOXES_LAYER {
            self.box_iter(index)
        } else {
            for cl in &self.custom_layers {
                if cl.0 == layer {
                    return self.custom_region_iter(layer, index);
                }
            }
            panic!("Invalid region layer for StdOverlay: {}", layer.name())
        }
    }

    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        if i1[0] == i2[0] || i1[1] == i2[1] {
            return true;
        }
        let (b1, _) = self.to_box_coords(i1);
        let (b2, _) = self.to_box_coords(i2);
        if b1 == b2 { return true; }
        if self.custom_layers.is_empty() {
            return false;
        }
        for cl in &self.custom_layers {
            for region in &cl.1 {
                if region.1.contains(&i1) && region.1.contains(&i2) {
                    return true;
                }
            }
        }
        false
    }

    fn parse_state<V: Value>(&self, s: &str) -> Result<State<V, Self>, Error> {
        let parsed = parse_val_grid::<V>(s, N, M)?;
        State::<V, Self>::with_givens(self.clone(), parsed)
    }

    fn serialize_state<V: Value>(&self, s: &State<V, Self>) -> String {
        let mut result = String::new();
        for r in 0..N {
            for c in 0..M {
                if let Some(v) = s.get([r, c]) {
                    result.push_str(v.to_string().as_str())
                } else {
                    result.push('.');
                }
            }
            result.push('\n');
        }
        result
    }
}

pub fn nine_standard_overlay() -> StdOverlay<9, 9> {
    StdOverlay::new(3, 3, 3, 3)
}
pub fn eight_standard_overlay() -> StdOverlay<8, 8> {
    StdOverlay::new(4, 2, 2, 4)
}
pub fn six_standard_overlay() -> StdOverlay<6, 6> {
    StdOverlay::new(3, 2, 2, 3)
}
pub fn four_standard_overlay() -> StdOverlay<4, 4> {
    StdOverlay::new(2, 2, 2, 2)
}

pub const ROW_CONFLICT_ATTRIBUTION: &str = "ROW_CONFLICT";
pub const COL_CONFLICT_ATTRIBUTION: &str = "COL_CONFLICT";
pub const BOX_CONFLICT_ATTRIBUTION: &str = "BOX_CONFLICT";

pub struct StdChecker<const N: usize, const M: usize, const MIN: u8, const MAX: u8> {
    overlay: StdOverlay<N, M>,
    row: [VBitSet<StdVal<MIN, MAX>>; N],
    col: [VBitSet<StdVal<MIN, MAX>>; M],
    boxes: Box<[VBitSet<StdVal<MIN, MAX>>]>,
    row_attr: Key<Attribution>,
    col_attr: Key<Attribution>,
    box_attr: Key<Attribution>,
    illegal: Option<(Index, StdVal<MIN, MAX>, Key<Attribution>)>,
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8> StdChecker<N, M, MIN, MAX> {
    pub fn new(overlay: &StdOverlay<N, M>) -> Self {
        return Self {
            overlay: overlay.clone(),
            row: std::array::from_fn(|_| VBitSet::<StdVal<MIN, MAX>>::full()),
            col: std::array::from_fn(|_| VBitSet::<StdVal<MIN, MAX>>::full()),
            boxes: vec![VBitSet::<StdVal<MIN, MAX>>::full(); overlay.boxes()].into_boxed_slice(),
            row_attr: Key::register(ROW_CONFLICT_ATTRIBUTION),
            col_attr: Key::register(COL_CONFLICT_ATTRIBUTION),
            box_attr: Key::register(BOX_CONFLICT_ATTRIBUTION),
            illegal: None,
        }
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8>
Debug for StdChecker<N, M, MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((i, v, a)) = &self.illegal {
            write!(f, "Illegal move: {:?}={:?} ({})\n", i, v, a.name())?;
        }
        write!(f, "Unused vals by row:\n")?;
        for r in 0..N {
            write!(f, " {}: {}\n", r, self.row[r].to_string())?;
        }
        write!(f, "Unused vals by col:\n")?;
        for c in 0..M {
            write!(f, " {}: {}\n", c, self.col[c].to_string())?;
        }
        write!(f, "Unused vals by box:\n")?;
        for b in 0..self.overlay.boxes() {
            write!(f, " {}: {}\n", b, self.boxes[b].to_string())?;
        }
        Ok(())
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8>
Stateful<StdVal<MIN, MAX>> for StdChecker<N, M, MIN, MAX> {
    fn reset(&mut self) {
        self.row = std::array::from_fn(|_| VBitSet::<StdVal<MIN, MAX>>::full());
        self.col = std::array::from_fn(|_| VBitSet::<StdVal<MIN, MAX>>::full());
        self.boxes = vec![VBitSet::<StdVal<MIN, MAX>>::full(); self.overlay.boxes()].into_boxed_slice();
        self.illegal = None;
    }

    fn apply(&mut self, index: Index, value: StdVal<MIN, MAX>) -> Result<(), Error> {
        // In theory we could be allow multiple illegal moves and just
        // invalidate and recalculate the grid or something, but it seems hard.
        if self.illegal.is_some() {
            return Err(ILLEGAL_ACTION_RC);
        }
        let (b, _) = self.overlay.to_box_coords(index);
        if !self.row[index[0]].contains(&value) {
            self.illegal = Some((index, value, self.row_attr));
            return Ok(());
        } else if !self.col[index[1]].contains(&value) {
            self.illegal = Some((index, value, self.col_attr));
            return Ok(());
        } else if !self.boxes[b].contains(&value){
            self.illegal = Some((index, value, self.box_attr));
            return Ok(());
        }
        self.row[index[0]].remove(&value);
        self.col[index[1]].remove(&value);
        self.boxes[b].remove(&value);
        Ok(())
    }

    fn undo(&mut self, index: Index, value: StdVal<MIN, MAX>) -> Result<(), Error> {
        if let Some((i, v, _)) = self.illegal {
            if i != index || v != value {
                return Err(UNDO_MISMATCH_ERROR);
            } else {
                self.illegal = None;
                return Ok(());
            }
        }
        let (b, _) = self.overlay.to_box_coords(index);
        self.row[index[0]].insert(&value);
        self.col[index[1]].insert(&value);
        self.boxes[b].insert(&value);
        Ok(())
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8>
Constraint<StdVal<MIN, MAX>, StdOverlay<N, M>> for StdChecker<N, M, MIN, MAX> {
    fn name(&self) -> Option<String> { Some("StdChecker".to_string()) }
    fn check(&self, puzzle: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, ranking: &mut RankingInfo<StdVal<MIN, MAX>>) -> ConstraintResult<StdVal<MIN, MAX>> {
        let grid = ranking.cells_mut();
        if let Some((_, _, a)) = &self.illegal {
            return ConstraintResult::Contradiction(*a);
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

    fn debug_at(&self, _: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, index: Index) -> Option<String> {
        let header = "StdChecker:\n";
        if let Some((i, v, a)) = &self.illegal {
            if *i == index {
                return Some(format!("{}  Illegal move: {} ({})", header, v, a.name()));
            }
        }
        let [r, c] = index;
        let (b, _) = self.overlay.to_box_coords(index);
        Some(format!(
            "{}  Unused vals in this row: {:?}\n  Unused vals in this col: {:?}\n  Unused vals in this box: {:?}",
            header,
            unpack_stdval_vals::<MIN, MAX, _>(&self.row[r]),
            unpack_stdval_vals::<MIN, MAX, _>(&self.col[c]),
            unpack_stdval_vals::<MIN, MAX, _>(&self.boxes[b]),
        ))
    }

    fn debug_highlight(&self, _: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, index: Index) -> Option<(u8, u8, u8)> {
        if let Some((i, _, _)) = &self.illegal {
            if *i == index {
                return Some((200, 0, 0));
            }
        }
        None
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ranker::Ranker;
    use crate::{core::test_util::replay_givens, ranker::StdRanker, solver::FindFirstSolution};
    use crate::core::test_util::round_trip_value;
    use crate::constraint::test_util::assert_contradiction;

    pub fn naive_min_max_sum(min: u8, max: u8, len: u8) -> Option<(u8, u8)> {
        if min + len - 1 > max {
            None
        } else {
            Some(((min..=(min+len-1)).sum(), ((max+1-len)..=max).sum()))
        }
    }

    #[test]
    fn test_min_max_sum() {
        for min in 1..=8 {
            for max in (min+1)..=9 {
                for len in 1..=9 {
                    assert_eq!(
                        min_max_sum(min, max, len),
                        naive_min_max_sum(min, max, len),
                        "for min_max_sum({}, {}, {})", min, max, len
                    );
                }
            }
        }
    }

    #[test]
    fn test_stdval() {
        // Closed interval, so it's 9-3+1
        assert_eq!(StdVal::<3, 9>::cardinality(), 7);
        // Values get serialized in the range 0..=(MAX-MIN)
        assert_eq!(StdVal::<3, 9>(3).to_uval(), UVal::new(0));
        assert_eq!(StdVal::<3, 9>(6).to_uval(), UVal::new(3));
        assert_eq!(StdVal::<3, 9>(9).to_uval(), UVal::new(6));
        // This round-trips
        assert_eq!(round_trip_value(StdVal::<3, 9>(3)).val(), 3);
        assert_eq!(round_trip_value(StdVal::<3, 9>(6)).val(), 6);
        assert_eq!(round_trip_value(StdVal::<3, 9>(9)).val(), 9);
    }

    #[test]
    fn test_stdval_set() {
        let mut mostly_empty = VBitSet::<StdVal<3, 9>>::empty();
        assert_eq!(unpack_stdval_vals::<3, 9, _>(&mostly_empty), Vec::<u8>::new());
        mostly_empty.insert(&StdVal::<3, 9>::new(4));
        assert_eq!(unpack_stdval_vals::<3, 9, _>(&mostly_empty), vec![4]);
        let mut mostly_full = VBitSet::<StdVal<3, 9>>::full();
        assert_eq!(
            unpack_stdval_vals::<3, 9, _>(&mostly_full),
            vec![3, 4, 5, 6, 7, 8, 9],
        );
        mostly_full.remove(&StdVal::<3, 9>::new(4));
        assert_eq!(
            unpack_stdval_vals::<3, 9, _>(&mostly_full),
            vec![3, 5, 6, 7, 8, 9],
        );
    }

    #[test]
    fn test_stdval_stats() {
        // Min=1, Max=4
        assert_eq!(stdval_sum_bound::<1, 4>(1), Some((1, 4)));
        // Min=1+2, Max=4+3
        assert_eq!(stdval_sum_bound::<1, 4>(2), Some((3, 7)));
        // Min=1+2+3, Max=4+3+2
        assert_eq!(stdval_sum_bound::<1, 4>(3), Some((6, 9)));
        // Min=Max=1+2+3+4
        assert_eq!(stdval_sum_bound::<1, 4>(4), Some((10, 10)));

        // 1
        assert_eq!(stdval_len_bound::<1, 4>(1), Some((1, 1)));
        // 2
        assert_eq!(stdval_len_bound::<1, 4>(2), Some((1, 1)));
        // 3, 1+2
        assert_eq!(stdval_len_bound::<1, 4>(3), Some((1, 2)));
        // 4, 1+3
        assert_eq!(stdval_len_bound::<1, 4>(4), Some((1, 2)));
        // 1+4, 2+3
        assert_eq!(stdval_len_bound::<1, 4>(5), Some((2, 2)));
        // 2+4, 1+2+3
        assert_eq!(stdval_len_bound::<1, 4>(6), Some((2, 3)));
        // 3+4, 1+2+4
        assert_eq!(stdval_len_bound::<1, 4>(7), Some((2, 3)));
        // 1+3+4
        assert_eq!(stdval_len_bound::<1, 4>(8), Some((3, 3)));
        // 2+3+4
        assert_eq!(stdval_len_bound::<1, 4>(9), Some((3, 3)));
        // 1+2+3+4
        assert_eq!(stdval_len_bound::<1, 4>(10), Some((4, 4)));
    }

    #[test]
    fn test_sudoku_grid() {
        let mut sudoku = nine_standard_empty();
        assert_eq!(sudoku.apply([0, 0], StdVal(5)), Ok(()));
        assert_eq!(sudoku.apply([8, 8], StdVal(1)), Ok(()));
        assert_eq!(sudoku.get([0, 0]), Some(StdVal(5)));
        assert_eq!(sudoku.undo([0, 0], StdVal(5)), Ok(()));
        assert_eq!(sudoku.get([0, 0]), None);
        assert_eq!(sudoku.get([8, 8]), Some(StdVal(1)));
        sudoku.reset();
        assert_eq!(sudoku.get([8, 8]), None);
    }

    #[test]
    fn test_sudoku_overlay() {
        let overlay = nine_standard_overlay();
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

    fn apply2<V: Value>(s1: &mut dyn Stateful<V>, s2: &mut dyn Stateful<V>, index: Index, value: V) {
        s1.apply(index, value).unwrap();
        s2.apply(index, value).unwrap();
    }

    #[test]
    fn test_sudoku_row_violation() {
        let mut sudoku = nine_standard_empty();
        let mut checker = StdChecker::new(sudoku.overlay());
        apply2(&mut sudoku, &mut checker, [5, 3], StdVal(1));
        apply2(&mut sudoku, &mut checker, [5, 4], StdVal(3));
        let mut ranking = StdRanker::default().init_ranking(&sudoku);
        match checker.check(&sudoku, &mut ranking) {
            ConstraintResult::Contradiction(a) => panic!("Unexpected contradiction: {}", a.name()),
            _ => {},
        };
        apply2(&mut sudoku, &mut checker, [5, 8], StdVal(1));
        assert_contradiction(checker.check(&sudoku, &mut ranking), "ROW_CONFLICT");
    }

    #[test]
    fn test_sudoku_col_violation() {
        let mut sudoku = nine_standard_empty();
        let mut checker = StdChecker::new(sudoku.overlay());
        apply2(&mut sudoku, &mut checker, [1, 3], StdVal(2));
        apply2(&mut sudoku, &mut checker, [3, 3], StdVal(7));
        let mut ranking = StdRanker::default().init_ranking(&sudoku);
        match checker.check(&sudoku, &mut ranking) {
            ConstraintResult::Contradiction(a) => panic!("Unexpected contradiction: {}", a.name()),
            _ => {},
        };
        apply2(&mut sudoku, &mut checker, [6, 3], StdVal(2));
        assert_contradiction(checker.check(&sudoku, &mut ranking), "COL_CONFLICT");
    }

    #[test]
    fn test_sudoku_box_violation() {
        let mut sudoku = nine_standard_empty();
        let mut checker = StdChecker::new(sudoku.overlay());
        apply2(&mut sudoku, &mut checker, [3, 0], StdVal(8));
        apply2(&mut sudoku, &mut checker, [4, 1], StdVal(2));
        let mut ranking = StdRanker::default().init_ranking(&sudoku);
        match checker.check(&sudoku, &mut ranking) {
            ConstraintResult::Contradiction(a) => panic!("Unexpected contradiction: {}", a.name()),
            _ => {},
        };
        apply2(&mut sudoku, &mut checker, [5, 2], StdVal(8));
        assert_contradiction(checker.check(&sudoku, &mut ranking), "BOX_CONFLICT");
    }

    #[test]
    fn test_sudoku_parse() {
        let input: &str = "5 . 3|. . .|. . .\n\
                           6 . .|1 9 5|. . .\n\
                           . 9 8|. . .|. 6 .\n\
                           -----+-----+-----\n\
                           8 . .|. 6 .|. . 3\n\
                           4 . .|8 . 3|. . 1\n\
                           7 . .|. 2 .|. . 6\n\
                           -----+-----+-----\n\
                           . 6 .|. . .|2 8 .\n\
                           . . .|4 1 9|. . 5\n\
                           . . .|. . .|8 . 9\n";
        let mut sudoku = nine_standard_parse(input).unwrap();
        replay_givens(&mut sudoku);
        assert_eq!(sudoku.get([0, 0]), Some(StdVal::new(5)));
        assert_eq!(sudoku.get([8, 8]), Some(StdVal::new(9)));
        assert_eq!(sudoku.get([2, 7]), Some(StdVal::new(6)));
        assert_eq!(sudoku.overlay().serialize_pretty(&sudoku), input);
    }

    #[test]
    fn test_sudoku_reset_remembers_givens() {
        let input: &str = "5 . 3|. . .|. . .\n\
                           6 . .|1 9 5|. . .\n\
                           . 9 8|. . .|. 6 .\n\
                           -----+-----+-----\n\
                           8 . .|. 6 .|. . 3\n\
                           4 . .|8 . 3|. . 1\n\
                           7 . .|. 2 .|. . 6\n\
                           -----+-----+-----\n\
                           . 6 .|. . .|2 8 .\n\
                           . . .|4 1 9|. . 5\n\
                           . . .|. . .|8 . 9\n";
        let mut sudoku = nine_standard_parse(input).unwrap();
        replay_givens(&mut sudoku);
        assert_eq!(sudoku.get([0, 1]), None);
        sudoku.apply([0, 1], StdVal::new(2)).unwrap();
        assert_eq!(sudoku.get([0, 1]), Some(StdVal::new(2)));
        sudoku.reset();
        replay_givens(&mut sudoku);
        assert_eq!(sudoku.get([0, 1]), None);
        assert_eq!(sudoku.overlay().serialize_pretty(&sudoku), input);
    }

    #[test]
    fn test_nine_solve() {
        // #t1d1p1 from sudoku-puzzles.net
        let input: &str = ". 7 .|5 8 3|. 2 .\n\
                           . 5 9|2 . .|3 . .\n\
                           3 4 .|. . 6|5 . 7\n\
                           -----+-----+-----\n\
                           7 9 5|. . .|6 3 2\n\
                           . . 3|6 9 7|1 . .\n\
                           6 8 .|. . 2|7 . .\n\
                           -----+-----+-----\n\
                           9 1 4|8 3 5|. 7 6\n\
                           . 3 .|7 . 1|4 9 5\n\
                           5 6 7|4 2 9|. 1 3\n";
        let ranker = StdRanker::default();
        let mut sudoku = nine_standard_parse(input).unwrap();
        let mut checker = StdChecker::new(sudoku.overlay());
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &ranker, &mut checker, None);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.state().get([2, 2]), Some(StdVal::new(2)));
                assert_eq!(solved.state().get([2, 3]), Some(StdVal::new(9)));
                assert_eq!(solved.state().get([2, 4]), Some(StdVal::new(1)));
            }
            Err(e) => panic!("Failed to solve sudoku: {:?}", e),
        }
    }

    #[test]
    fn test_eight_solve() {
        // #t34d1p1 from sudoku-puzzles.net
        let input: &str = "2 .|. .|1 .|3 8\n\
                           3 1|6 .|. 7|. 2\n\
                           ---+---+---+---\n\
                           . 4|5 .|. .|8 .\n\
                           1 .|. 2|6 4|7 5\n\
                           ---+---+---+---\n\
                           . .|4 7|5 .|. .\n\
                           5 2|. .|7 .|6 .\n\
                           ---+---+---+---\n\
                           . 7|1 3|. .|. 6\n\
                           4 6|. .|8 .|. .\n";
        let ranker = StdRanker::default();
        let mut sudoku = eight_standard_parse(input).unwrap();
        let mut checker = StdChecker::new(sudoku.overlay());
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &ranker, &mut checker, None);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.state().get([6, 4]), Some(StdVal::new(2)));
                assert_eq!(solved.state().get([6, 5]), Some(StdVal::new(5)));
                assert_eq!(solved.state().get([6, 6]), Some(StdVal::new(4)));
            }
            Err(e) => panic!("Failed to solve sudoku: {:?}", e),
        }
    }

    #[test]
    fn test_six_solve() {
        // #t2d1p1 from sudoku-puzzles.net
        let input: &str = ". 3 .|4 . .\n\
                           . . 5|6 . 3\n\
                           -----+-----\n\
                           . . .|1 . .\n\
                           . 1 .|3 . 5\n\
                           -----+-----\n\
                           . 6 4|. 3 1\n\
                           . . 1|. 4 6\n";
        let ranker = StdRanker::default();
        let mut sudoku = six_standard_parse(input).unwrap();
        let mut checker = StdChecker::new(sudoku.overlay());
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &ranker, &mut checker, None);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.state().get([2, 0]), Some(StdVal::new(6)));
                assert_eq!(solved.state().get([2, 1]), Some(StdVal::new(5)));
                assert_eq!(solved.state().get([2, 2]), Some(StdVal::new(3)));
            }
            Err(e) => panic!("Failed to solve sudoku: {:?}", e),
        }
    }

    #[test]
    fn test_four_solve() {
        // #t14d1p1 from sudoku-puzzles.net
        let input: &str = ". .|. 4\n\
                           . .|. .\n\
                           ---+---\n\
                           2 .|. 3\n\
                           4 .|1 2\n";
        let ranker = StdRanker::default();
        let mut sudoku = four_standard_parse(input).unwrap();
        let mut checker = StdChecker::new(sudoku.overlay());
        let mut finder = FindFirstSolution::new(
            &mut sudoku, &ranker, &mut checker, None);
        match finder.solve() {
            Ok(solution) => {
                assert!(solution.is_some());
                let solved = solution.unwrap();
                assert_eq!(solved.state().get([0, 0]), Some(StdVal::new(1)));
                assert_eq!(solved.state().get([0, 1]), Some(StdVal::new(2)));
                assert_eq!(solved.state().get([0, 2]), Some(StdVal::new(3)));
            }
            Err(e) => panic!("Failed to solve sudoku: {:?}", e),
        }
    }
}