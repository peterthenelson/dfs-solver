use std::marker::PhantomData;
use crate::core::{empty_map, empty_set, readable_feature, unpack_values, Attribution, BranchPoint, CertainDecision, ConstraintResult, DecisionGrid, FVMaybeNormed, FVNormed, FeatureKey, FeatureVec, Index, State, UVMap, UVSet, Value, WithId};
use crate::sudoku::{Overlay, SState, SVal};

/// A ranker finds the "best" place in the grid to make a guess. In theory, we
/// could extend this to return multiple guesses, but since a given index
/// provides a mutually exclusive and exhaustive set of guesses, there isn't
/// really a need.
pub trait Ranker<V: Value, S: State<V>> {
    // Note: the ranker must not suggest already filled cells.
    fn top(&self, grid: &DecisionGrid<V>, puzzle: &S) -> BranchPoint<V>;

    // Collapse a DecisionGrid into a ConstraintResult, returning any Certainty
    // or Contradiction that is present. This must be compatible with top() --
    // i.e., top() must always return something possible if no Contradiction is
    // found here.
    fn to_constraint_result(&self, grid: &DecisionGrid<V>, puzzle: &S) -> ConstraintResult<V>;
}

pub const NUM_POSSIBLE_FEATURE: &str = "NUM_POSSIBLE";
pub const DG_EMPTY_ATTRIBUTION: &str = "DG_EMPTY";
pub const DG_TOP_CELL_ATTRIBUTION: &str = "DG_CELL_TOP";
pub const DG_NO_VALS_ATTRIBUTION: &str = "DG_CELL_NO_VALS";
pub const DG_ONE_VAL_ATTRIBUTION: &str = "DG_CELL_ONE_VAL";
pub const DG_TOP_VAL_ATTRIBUTION: &str = "DG_VAL_TOP";
pub const DG_NO_CELLS_ATTRIBUTION: &str = "DG_VAL_NO_CELLS";
pub const DG_ONE_CELL_ATTRIBUTION: &str = "DG_VAL_ONE_CELL";

/// A linear scorer. Note that NUM_POSSIBLE is the (most important!) feature
/// that indicates how many possible values are left for a cell.
pub struct LinearRanker {
    weights: FeatureVec<FVNormed>,
    num_possible: FeatureKey<WithId>,
    empty_attribution: Attribution<WithId>,
    top_cell_attribution: Attribution<WithId>,
    no_vals_attribution: Attribution<WithId>,
    one_val_attribution: Attribution<WithId>,
}

impl LinearRanker {
    pub fn new(feature_weights: FeatureVec<FVMaybeNormed>) -> Self {
        let mut weights = feature_weights.clone();
        weights.normalize(|id, _, _| panic!("Duplicate feature in weights vec: {:?}", readable_feature(id)));
        LinearRanker {
            weights: weights.try_normalized().unwrap().clone(),
            num_possible: FeatureKey::new(NUM_POSSIBLE_FEATURE).unwrap(),
            empty_attribution: Attribution::new(DG_EMPTY_ATTRIBUTION).unwrap(),
            top_cell_attribution: Attribution::new(DG_TOP_CELL_ATTRIBUTION).unwrap(),
            no_vals_attribution: Attribution::new(DG_NO_VALS_ATTRIBUTION).unwrap(),
            one_val_attribution: Attribution::new(DG_ONE_VAL_ATTRIBUTION).unwrap(),
        }
    }

    // {NUM_POSSIBLE: -100} I.e., highly prioritize cells with the fewest
    // possible values.
    pub fn default() -> Self {
        let mut weights = FeatureVec::new();
        weights.add(&FeatureKey::new(NUM_POSSIBLE_FEATURE).unwrap(), -100.0);
        Self::new(weights)
    }
}

impl <V: Value, S: State<V>> Ranker<V, S> for LinearRanker {
    fn top(&self, grid: &DecisionGrid<V>, puzzle: &S) -> BranchPoint<V> {
        let mut top_index = None;
        let mut top_score: f64 = 0.0;
        for r in 0..grid.rows() {
            for c in 0..grid.cols() {
                if puzzle.get([r, c]).is_some() {
                    continue;
                }
                let g = grid.get([r, c]);
                let mut fv = g.1.clone();
                fv.add(&self.num_possible, g.0.len() as f64);
                let score = fv.normalize_and(|_, a, b| a+b).dot_product(&self.weights);
                if top_index.is_none() || score > top_score {
                    top_score = score;
                    top_index = Some([r, c]);
                }
            }
        }
        if let Some(index) = top_index {
            BranchPoint::for_cell(0, self.top_cell_attribution.clone(), index, unpack_values(&grid.get(index).0))
        } else {
            BranchPoint::empty(0, self.empty_attribution.clone())
        }
    }

    fn to_constraint_result(&self, grid: &DecisionGrid<V>, puzzle: &S) -> ConstraintResult<V> {
        for r in 0..grid.rows() {
            for c in 0..grid.cols() {
                if puzzle.get([r, c]).is_none() {
                    let cell = &grid.get([r, c]).0;
                    if cell.len() == 0 {
                        return ConstraintResult::Contradiction(self.no_vals_attribution.clone());
                    } else if cell.len() == 1 {
                        let v = unpack_values::<V>(cell)[0];
                        return ConstraintResult::Certainty(
                            CertainDecision::new([r, c], v),
                            self.one_val_attribution.clone(),
                        );
                    }
                }
            }
        }
        ConstraintResult::Ok
    }
}

/// A linear scorer that also suggests possibilities based on the overlay (e.g.,
/// the 9 in the 3rd box goes in index A or B) rather than just based on cells.
/// Note that NUM_POSSIBLE is the (most important!) feature that indicates how
/// many possible indices are left for a particular value in a region.
pub struct OverlaySensitiveLinearRanker {
    weights: FeatureVec<FVNormed>,
    num_possible: FeatureKey<WithId>,
    combinator: fn (usize, f64, f64) -> f64,
    empty_attribution: Attribution<WithId>,
    top_cell_attribution: Attribution<WithId>,
    top_val_attribution: Attribution<WithId>,
    no_vals_attribution: Attribution<WithId>,
    one_val_attribution: Attribution<WithId>,
    no_cells_attribution: Attribution<WithId>,
    one_cell_attribution: Attribution<WithId>,
}

pub struct OSLRRegionInfo<V: Value> {
    // Values that have already been filled into the puzzle.
    filled: UVSet<V::U>,
    // Cells that a given value can go into.
    cell_choices: UVMap<V::U, Vec<Index>>,
    // Feature vectors for a given value.
    feature_vecs: UVMap<V::U, FeatureVec<FVMaybeNormed>>,
    p_v_: PhantomData<V>,
}

impl <V: Value> OSLRRegionInfo<V> {
    pub fn new() -> Self {
        Self {
            filled: empty_set::<V>(),
            cell_choices: empty_map::<V, Vec<Index>>(),
            feature_vecs: empty_map::<V, FeatureVec<FVMaybeNormed>>(),
            p_v_: PhantomData,
        }
    }
}

impl OverlaySensitiveLinearRanker {
    pub fn new(feature_weights: FeatureVec<FVMaybeNormed>, combine_features: fn (usize, f64, f64) -> f64) -> Self {
        let mut weights = feature_weights.clone();
        weights.normalize(|id, _, _| panic!("Duplicate feature in weights vec: {:?}", readable_feature(id)));
        OverlaySensitiveLinearRanker {
            weights: weights.try_normalized().unwrap().clone(),
            num_possible: FeatureKey::new(NUM_POSSIBLE_FEATURE).unwrap(),
            combinator: combine_features,
            empty_attribution: Attribution::new(DG_EMPTY_ATTRIBUTION).unwrap(),
            top_cell_attribution: Attribution::new(DG_TOP_CELL_ATTRIBUTION).unwrap(),
            top_val_attribution: Attribution::new(DG_TOP_VAL_ATTRIBUTION).unwrap(),
            no_vals_attribution: Attribution::new(DG_NO_VALS_ATTRIBUTION).unwrap(),
            one_val_attribution: Attribution::new(DG_ONE_VAL_ATTRIBUTION).unwrap(),
            no_cells_attribution: Attribution::new(DG_NO_CELLS_ATTRIBUTION).unwrap(),
            one_cell_attribution: Attribution::new(DG_ONE_CELL_ATTRIBUTION).unwrap(),
        }
    }

    // {NUM_POSSIBLE: -100} I.e., highly prioritize cells with the fewest
    // possible values. Default behavior for combining features across the
    // indices in a layout-region is to take the maximum.
    pub fn default() -> Self {
        let mut weights = FeatureVec::new();
        weights.add(&FeatureKey::new(NUM_POSSIBLE_FEATURE).unwrap(), -100.0);
        Self::new(weights, |_, a, b| f64::max(a, b))
    }

    /// Exposes how the ranker looks at the grid when calculating
    /// ::ValueInRegion candidates. Useful for debugging without needing to
    /// duplicate the implementation of the ranker.
    pub fn region_info<const N: usize, const M: usize, const MIN: u8, const MAX: u8, O: Overlay>(
        &self, grid: &DecisionGrid<SVal<MIN, MAX>>, puzzle: &SState<N, M, MIN, MAX, O>, dim: usize, p: usize,
    ) -> OSLRRegionInfo<SVal<MIN, MAX>> {
        let mut info = OSLRRegionInfo::new();
        for index in puzzle.get_overlay().partition_iter(dim, p) {
            if let Some(val) = puzzle.get(index) {
                info.filled.insert(val.to_uval());
                continue;
            }
            let g = grid.get(index);
            for uv in g.0.iter() {
                info.cell_choices.get_mut(uv).push(index);
                info.feature_vecs.get_mut(uv).extend(&g.1);
            }
        }
        for v in SVal::<MIN, MAX>::possibilities() {
            let uv = v.to_uval();
            if info.filled.contains(uv) {
                continue;
            }
            // TODO: Should we normalize the other feature values by
            // 1/alternatives? Otherwise we're implicitly overweighting
            // towards choosing a ::ValueInRegion over a ::Cell.
            info.feature_vecs.get_mut(uv).add(
                &self.num_possible,
                info.cell_choices.get(uv).len() as f64,
            );
        }
        info
    }
}

enum OSLRChoice<const MIN: u8, const MAX: u8> {
    Cell(Index),
    ValueInRegion(SVal<MIN, MAX>, Vec<Index>),
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8, O: Overlay>
Ranker<SVal<MIN, MAX>, SState<N, M, MIN, MAX, O>> for OverlaySensitiveLinearRanker {
    fn top(&self, grid: &DecisionGrid<SVal<MIN, MAX>>, puzzle: &SState<N, M, MIN, MAX, O>) -> BranchPoint<SVal<MIN, MAX>> {
        let mut top_choice = None;
        let mut top_score: f64 = 0.0;
        for r in 0..grid.rows() {
            for c in 0..grid.cols() {
                if puzzle.get([r, c]).is_some() {
                    continue;
                }
                let g = grid.get([r, c]);
                let mut fv = g.1.clone();
                fv.add(&self.num_possible, g.0.len() as f64);
                let score = fv.normalize_and(|_, a, b| a+b).dot_product(&self.weights);
                if top_choice.is_none() || score > top_score {
                    top_score = score;
                    top_choice = Some(OSLRChoice::Cell([r, c]));
                }
            }
        }
        let overlay = puzzle.get_overlay();
        for dim in 0..overlay.partition_dimension() {
            for p in 0..overlay.n_partitions(dim) {
                let mut info = self.region_info(grid, puzzle, dim, p);
                for v in SVal::<MIN, MAX>::possibilities() {
                    let uv = v.to_uval();
                    if info.filled.contains(uv) {
                        continue;
                    }
                    let score = info.feature_vecs.get_mut(uv).normalize_and(self.combinator).dot_product(&self.weights);
                    if top_choice.is_none() || score > top_score {
                        top_score = score;
                        top_choice = Some(OSLRChoice::ValueInRegion(v, info.cell_choices.get(uv).clone()));
                    }
                }
            }
        }
        match top_choice {
            Some(OSLRChoice::Cell(index)) => {
                BranchPoint::for_cell(0, self.top_cell_attribution.clone(), index, unpack_values(&grid.get(index).0))
            },
            Some(OSLRChoice::ValueInRegion(val, alternatives)) => {
                BranchPoint::for_value(0, self.top_val_attribution.clone(), val, alternatives)
            },
            None => BranchPoint::empty(0, self.empty_attribution.clone()),
        }
    }

    fn to_constraint_result(&self, grid: &DecisionGrid<SVal<MIN, MAX>>, puzzle: &SState<N, M, MIN, MAX, O>) -> ConstraintResult<SVal<MIN, MAX>> {
        for r in 0..grid.rows() {
            for c in 0..grid.cols() {
                if puzzle.get([r, c]).is_none() {
                    let cell = &grid.get([r, c]).0;
                    if cell.len() == 0 {
                        return ConstraintResult::Contradiction(self.no_vals_attribution.clone())
                    } else if cell.len() == 1 {
                        let v = unpack_values::<SVal<MIN, MAX>>(cell)[0];
                        return ConstraintResult::Certainty(
                            CertainDecision::new([r, c], v),
                            self.one_val_attribution.clone(),
                        );
                    }
                }
            }
        }
        let overlay = puzzle.get_overlay();
        for dim in 0..overlay.partition_dimension() {
            for p in 0..overlay.n_partitions(dim) {
                let mut count = empty_map::<SVal<MIN, MAX>, usize>();
                let mut first_index = empty_map::<SVal<MIN, MAX>, Option<Index>>();
                for index in overlay.partition_iter(dim, p) {
                    if let Some(val) = puzzle.get(index) {
                        *count.get_mut(val.to_uval()) += 1;
                        // Don't put anything into first_index, since this is
                        // already filled, not a certaint decision yet to be
                        // made.
                    } else {
                        let g = grid.get(index);
                        for uv in g.0.iter() {
                            *count.get_mut(uv) += 1;
                            let i = first_index.get_mut(uv);
                            if i.is_none() {
                                *i = Some(index);
                            }
                        }
                    }
                }
                for v in SVal::<MIN, MAX>::possibilities() {
                    let uv = v.to_uval();
                    let c = count.get(uv);
                    if *c == 0 {
                        return ConstraintResult::Contradiction(self.no_cells_attribution.clone());
                    } else if *c == 1 && first_index.get(uv).is_some() {
                        return ConstraintResult::Certainty(
                            CertainDecision::new(first_index.get(uv).unwrap(), v),
                            self.one_cell_attribution.clone(),
                        );
                    }
                }
            }
        }
        ConstraintResult::Ok
    }
}
