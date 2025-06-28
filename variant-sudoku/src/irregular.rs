use std::{collections::{HashMap, HashSet}, fmt::Debug};
use crate::{constraint::Constraint, core::{Attribution, ConstraintResult, DecisionGrid, Error, Index, Key, Overlay, RankingInfo, RegionLayer, State, Stateful, Unscored, VBitSet, VSet, VSetMut, Value, BOXES_LAYER, COLS_LAYER, ROWS_LAYER}, index_util::{parse_region_grid, parse_val_grid}};

#[derive(Debug, Clone)]
pub struct IrregularOverlay<const N: usize, const M: usize> {
    regions: Vec<Vec<Index>>,
    index_to_region_and_offset: HashMap<Index, (usize, usize)>,
}

impl <const N: usize, const M: usize> IrregularOverlay<N, M> {
    pub fn from_regions(regions: Vec<Vec<Index>>) -> Result<Self, Error> {
        if regions.is_empty() {
            return Err(Error::new(
                "Must have at least one region".to_string(),
            ))
        }
        let mut covered = HashSet::new();
        let mut region_size = None;
        let mut index_to_region_and_offset = HashMap::new();
        for (bi, region) in regions.iter().enumerate() {
            if let Some(rs) = region_size {
                if region.len() != rs {
                    return Err(Error::new(format!(
                        "All regions must be the same size ({} != {})",
                        region.len(), rs,
                    )))
                }
            } else {
                region_size = Some(region.len())
            }
            for (bo, cell) in region.iter().enumerate() {
                if covered.contains(&cell) {
                    panic!("Multiple regions contain cell: {:?}\n", cell);
                }
                covered.insert(cell);
                index_to_region_and_offset.insert(*cell, (bi, bo));
            }
        }
        for r in 0..N {
            for c in 0..M {
                if !covered.contains(&[r, c]) {
                    return Err(Error::new(format!(
                        "Regions must cover whole {}x{} grid; {:?} not covered",
                        N, M, [r, c],
                    )));
                }
            }
        }
        Ok(Self { regions, index_to_region_and_offset })
    }

    pub fn from_grid(s: &str) -> Result<Self, Error> {
        let parsed = parse_region_grid(s, N, M)?;
        let mut sorted_keys = parsed.keys().collect::<Vec<_>>();
        sorted_keys.sort();
        let regions = sorted_keys.into_iter()
            .map(|k| parsed[k].clone()).collect();
        Self::from_regions(regions)
    }

    pub fn nth_in_box(&self, bi: usize, bo: usize) -> Index {
        self.regions[bi][bo]
    }
}

enum IrregularOverlayIteratorState {
    Row(usize, usize),
    Col(usize, usize),
    Box(usize, usize),
}

pub struct IrregularOverlayIterator<'a, const N: usize, const M: usize> {
    overlay: &'a IrregularOverlay<N, M>,
    state: IrregularOverlayIteratorState,
}

impl <'a, const N: usize, const M: usize> Iterator for IrregularOverlayIterator<'a, N, M> {
    type Item = Index;
    fn next(&mut self) -> Option<Self::Item> {
        let ret: Index;
        let (rows, cols) = self.overlay.grid_dims();
        match self.state {
            IrregularOverlayIteratorState::Row(r, c) => {
                if c >= cols {
                    return None;
                }
                ret = [r, c];
                self.state = IrregularOverlayIteratorState::Row(r, c+1);
            },
            IrregularOverlayIteratorState::Col(c, r) => {
                if r >= rows {
                    return None;
                }
                ret = [r, c];
                self.state = IrregularOverlayIteratorState::Col(c, r+1);
            },
            IrregularOverlayIteratorState::Box(bi, bo) => {
                if bo >= self.overlay.cells_in_region(BOXES_LAYER, bi) {
                    return None
                }
                ret = self.overlay.nth_in_box(bi, bo);
                self.state = IrregularOverlayIteratorState::Box(bi, bo+1)
            },
        }
        Some(ret)
    }
}

impl <const N: usize, const M: usize> Overlay for IrregularOverlay<N, M> {
    type Iter<'a> = IrregularOverlayIterator<'a, N, M> where Self: 'a;

    fn grid_dims(&self) -> (usize, usize) { (N, M) }

    fn full_decision_grid<V: Value>(&self) -> DecisionGrid<V, Unscored> {
        DecisionGrid::full(N, M)
    }

    fn region_layers(&self) -> Vec<Key<RegionLayer>> {
        vec![ROWS_LAYER, COLS_LAYER, BOXES_LAYER]
    }
    
    fn regions_in_layer(&self, layer: Key<RegionLayer>) -> usize {
        let id = layer.id();
        if id == ROWS_LAYER.id() {
            N
        } else if id == COLS_LAYER.id() {
            M
        } else if id == BOXES_LAYER.id() {
            self.regions.len()
        } else {
            panic!("Invalid region layer for IrregularOverlay: {}", layer.name())
        }
    }

    fn cells_in_region(&self, layer: Key<RegionLayer>, _: usize) -> usize {
        let id = layer.id();
        if id == ROWS_LAYER.id() {
            M
        } else if id == COLS_LAYER.id() {
            N
        } else if id == BOXES_LAYER.id() {
            self.regions[0].len()
        } else {
            panic!("Invalid region layer for IrregularOverlay: {}", layer.name())
        }
    }

    fn enclosing_region_and_offset(&self, layer: Key<RegionLayer>, index: Index) -> Option<(usize, usize)> {
        let id = layer.id();
        if id == ROWS_LAYER.id() {
            Some((index[0], index[1]))
        } else if id == COLS_LAYER.id() {
            Some((index[1], index[0]))
        } else if id == BOXES_LAYER.id() {
            self.index_to_region_and_offset.get(&index).copied()
        } else {
            panic!("Invalid region layer for IrregularOverlay: {}", layer.name())
        }
    }
    
    fn region_iter(&self, layer: Key<RegionLayer>, index: usize) -> Self::Iter<'_> {
        let id = layer.id();
        if id == ROWS_LAYER.id() {
            IrregularOverlayIterator {
                overlay: self, 
                state: IrregularOverlayIteratorState::Row(index, 0),
            }
        } else if id == COLS_LAYER.id() {
            IrregularOverlayIterator {
                overlay: self, 
                state: IrregularOverlayIteratorState::Col(index, 0),
            }
        } else if id == BOXES_LAYER.id() {
            IrregularOverlayIterator {
                overlay: self, 
                state: IrregularOverlayIteratorState::Box(index, 0),
            }
        } else {
            panic!("Invalid region layer for IrregularOverlay: {}", layer.name())
        }
    }

    fn nth_in_region(&self, layer: Key<RegionLayer>, index: usize, offset: usize) -> Option<Index> {
        let id = layer.id();
        if id == ROWS_LAYER.id() {
            if index < N && offset < M {
                Some([index, offset])
            } else {
                None
            }
        } else if id == COLS_LAYER.id() {
            if index < M && offset < N {
                Some([offset, index])
            } else {
                None
            }
        } else if id == BOXES_LAYER.id() {
            if index < self.regions.len() && offset < self.regions[0].len() {
                Some(self.nth_in_box(index, offset))
            } else {
                None
            }
        } else {
            panic!("Invalid region layer for IrregularOverlay: {}", layer.name())
        }
    }

    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        if i1[0] == i2[0] || i1[1] == i2[1] {
            return true;
        }
        self.index_to_region_and_offset[&i1].0 == self.index_to_region_and_offset[&i2].0
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

pub const UNDO_MISMATCH_ERROR: Error = Error::new_const("Undo value mismatch");
pub const ILLEGAL_ACTION_RC: Error = Error::new_const("A row/col violation already exists; can't apply further actions.");
pub const ILLEGAL_ACTION_REGION: Error = Error::new_const("A region violation already exists; can't apply further actions.");
pub const ROW_CONFLICT_ATTRIBUTION: &str = "ROW_CONFLICT";
pub const COL_CONFLICT_ATTRIBUTION: &str = "COL_CONFLICT";
pub const REGION_CONFLICT_ATTRIBUTION: &str = "REGION_CONFLICT";

pub struct IrregularChecker<const N: usize, const M: usize, V: Value> {
    overlay: IrregularOverlay<N, M>,
    row: [VBitSet<V>; N],
    col: [VBitSet<V>; M],
    region: Box<[VBitSet<V>]>,
    row_attr: Key<Attribution>,
    col_attr: Key<Attribution>,
    region_attr: Key<Attribution>,
    illegal: Option<(Index, V, Key<Attribution>)>,
}

impl <const N: usize, const M: usize, V: Value> IrregularChecker<N, M, V> {
    pub fn new(state: &State<V, IrregularOverlay<N, M>>) -> Self {
        return Self {
            overlay: state.overlay().clone(),
            row: std::array::from_fn(|_| VBitSet::<V>::full()),
            col: std::array::from_fn(|_| VBitSet::<V>::full()),
            region: vec![VBitSet::<V>::full(); state.overlay().regions_in_layer(BOXES_LAYER)].into_boxed_slice(),
            row_attr: Key::register(ROW_CONFLICT_ATTRIBUTION),
            col_attr: Key::register(COL_CONFLICT_ATTRIBUTION),
            region_attr: Key::register(REGION_CONFLICT_ATTRIBUTION),
            illegal: None,
        }
    }
}

impl <const N: usize, const M: usize, V: Value>
Debug for IrregularChecker<N, M, V> {
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
        write!(f, "Unused vals by region:\n")?;
        for reg in 0..self.overlay.regions_in_layer(BOXES_LAYER) {
            write!(f, " {}: {}\n", reg, self.region[reg].to_string())?;
        }
        Ok(())
    }
}

impl <const N: usize, const M: usize, V: Value>
Stateful<V> for IrregularChecker<N, M, V> {
    fn reset(&mut self) {
        self.row = std::array::from_fn(|_| VBitSet::<V>::full());
        self.col = std::array::from_fn(|_| VBitSet::<V>::full());
        self.region = vec![VBitSet::<V>::full(); self.overlay.regions_in_layer(BOXES_LAYER)].into_boxed_slice();
        self.illegal = None;
    }

    fn apply(&mut self, index: Index, value: V) -> Result<(), Error> {
        // In theory we could be allow multiple illegal moves and just
        // invalidate and recalculate the grid or something, but it seems hard.
        if self.illegal.is_some() {
            return Err(ILLEGAL_ACTION_RC);
        }
        let (reg, _) = self.overlay.enclosing_region_and_offset(BOXES_LAYER, index).unwrap();
        if !self.row[index[0]].contains(&value) {
            self.illegal = Some((index, value, self.row_attr));
            return Ok(());
        } else if !self.col[index[1]].contains(&value) {
            self.illegal = Some((index, value, self.col_attr));
            return Ok(());
        } else if !self.region[reg].contains(&value){
            self.illegal = Some((index, value, self.region_attr));
            return Ok(());
        }
        self.row[index[0]].remove(&value);
        self.col[index[1]].remove(&value);
        self.region[reg].remove(&value);
        Ok(())
    }

    fn undo(&mut self, index: Index, value: V) -> Result<(), Error> {
        if let Some((i, v, _)) = self.illegal {
            if i != index || v != value {
                return Err(UNDO_MISMATCH_ERROR);
            } else {
                self.illegal = None;
                return Ok(());
            }
        }
        let (reg, _) = self.overlay.enclosing_region_and_offset(BOXES_LAYER, index).unwrap();
        self.row[index[0]].insert(&value);
        self.col[index[1]].insert(&value);
        self.region[reg].insert(&value);
        Ok(())
    }
}

impl <const N: usize, const M: usize, V: Value>
Constraint<V, IrregularOverlay<N, M>> for IrregularChecker<N, M, V> {
    fn check(&self, puzzle: &State<V, IrregularOverlay<N, M>>, ranking: &mut RankingInfo<V, Unscored>) -> ConstraintResult<V> {
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
                let (reg, _) = self.overlay.enclosing_region_and_offset(BOXES_LAYER, [r, c]).unwrap();
                cell.0.intersect_with(&self.row[r]);
                cell.0.intersect_with(&self.col[c]);
                cell.0.intersect_with(&self.region[reg]);
            }
        }
        ConstraintResult::Ok
    }

    fn debug_at(&self, _: &State<V, IrregularOverlay<N, M>>, index: Index) -> Option<String> {
        let header = "IrregularChecker:\n";
        if let Some((i, v, a)) = &self.illegal {
            if *i == index {
                return Some(format!("{}  Illegal move: {} ({})", header, v, a.name()));
            }
        }
        let [r, c] = index;
        let (reg, _) = self.overlay.enclosing_region_and_offset(BOXES_LAYER, [r, c]).unwrap();
        Some(format!(
            "{}  Unused vals in this row: {}\n  Unused vals in this col: {}\n  Unused vals in this region: {}",
            header, self.row[r].to_string(), self.col[c].to_string(), self.region[reg].to_string(),
        ))
    }
}
