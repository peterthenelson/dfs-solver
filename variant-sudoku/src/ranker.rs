use std::marker::PhantomData;
use crate::core::{empty_map, empty_set, readable_feature, unpack_values, Attribution, BranchPoint, CertainDecision, ConstraintResult, DecisionGrid, FVMaybeNormed, FVNormed, FeatureKey, FeatureVec, Index, Overlay, State, UVMap, UVSet, Value, WithId};

/// A ranker finds the "best" place in the grid to make a guess. This could
/// either be a cell ("Here are the mutually exclusive and exhaustive
/// possible values that go in [0, 0]") or a value in a region ("here are the
/// mutually exclusive and exhaustive possible cells where the 9 in row 3 can
/// go"). In theory, the ranker can choose to order the values/cells in the
/// branch point, but the StdRanker does not do so, and the DecisionGrid doesn't
/// provide useful guidance to base such a decision on.
pub trait Ranker<V: Value, O: Overlay, S: State<V, O>> {
    // Note: the ranker must not suggest already filled cells.
    fn top(&self, grid: &DecisionGrid<V>, puzzle: &S) -> BranchPoint<V>;

    // Score for a particular feature vec. Exposed for debugging reasons.
    fn score(&self, fv: &mut FeatureVec<FVMaybeNormed>) -> f64;

    // Collapse a DecisionGrid into a ConstraintResult, returning any Certainty
    // or Contradiction that is present. This must be compatible with top() --
    // i.e., top() must always return something possible if no Contradiction is
    // found here.
    fn to_constraint_result(&self, grid: &DecisionGrid<V>, puzzle: &S) -> ConstraintResult<V>;

    /// Exposes how the ranker looks at the grid when calculating candidates
    /// in regions (i.e., those implied by positive_constraint). Useful for
    /// debugging without needing to duplicate the implementation of the ranker.
    /// Implementations may return None if they don't generate candidates in
    /// this way.
    fn region_info(
        &self, grid: &DecisionGrid<V>, puzzle: &S, dim: usize, p: usize,
    ) -> Option<RankerRegionInfo<V>>;
}

pub struct RankerRegionInfo<V: Value> {
    // Values that have already been filled into the puzzle.
    pub filled: UVSet<V::U>,
    // Cells that a given value can go into.
    pub cell_choices: UVMap<V::U, Vec<Index>>,
    // Feature vectors for a given value.
    pub feature_vecs: UVMap<V::U, FeatureVec<FVMaybeNormed>>,
    p_v_: PhantomData<V>,
}

pub const NUM_POSSIBLE_FEATURE: &str = "NUM_POSSIBLE";
pub const DG_EMPTY_ATTRIBUTION: &str = "DG_EMPTY";
pub const DG_TOP_CELL_ATTRIBUTION: &str = "DG_CELL_TOP";
pub const DG_NO_VALS_ATTRIBUTION: &str = "DG_CELL_NO_VALS";
pub const DG_ONE_VAL_ATTRIBUTION: &str = "DG_CELL_ONE_VAL";
pub const DG_TOP_VAL_ATTRIBUTION: &str = "DG_VAL_TOP";
pub const DG_NO_CELLS_ATTRIBUTION: &str = "DG_VAL_NO_CELLS";
pub const DG_ONE_CELL_ATTRIBUTION: &str = "DG_VAL_ONE_CELL";

/// A linear scorer that also suggests possibilities based on the cells in the
/// grid, as well as based on the overlay (e.g., the 9 in the 3rd box can only
/// go in index A or B). Note that NUM_POSSIBLE is the (most important!) feature
/// that indicates how many possible indices are left for a particular value in
/// a region.
pub struct StdRanker {
    positive_constraint: bool,
    weights: FeatureVec<FVNormed>,
    num_possible: FeatureKey<WithId>,
    combinator: fn (usize, f64, f64) -> f64,
    empty_attr: Attribution<WithId>,
    top_cell_attr: Attribution<WithId>,
    top_val_attr: Attribution<WithId>,
    no_vals_attr: Attribution<WithId>,
    one_val_attr: Attribution<WithId>,
    no_cells_attr: Attribution<WithId>,
    one_cell_attr: Attribution<WithId>,
}

impl <V: Value> RankerRegionInfo<V> {
    pub fn new() -> Self {
        Self {
            filled: empty_set::<V>(),
            cell_choices: empty_map::<V, Vec<Index>>(),
            feature_vecs: empty_map::<V, FeatureVec<FVMaybeNormed>>(),
            p_v_: PhantomData,
        }
    }
}

impl StdRanker {
    pub fn new(positive_constraint: bool, feature_weights: FeatureVec<FVMaybeNormed>, combine_features: fn (usize, f64, f64) -> f64) -> Self {
        let mut weights = feature_weights.clone();
        weights.normalize(|id, _, _| panic!("Duplicate feature in weights vec: {:?}", readable_feature(id)));
        StdRanker {
            positive_constraint,
            weights: weights.try_normalized().unwrap().clone(),
            num_possible: FeatureKey::new(NUM_POSSIBLE_FEATURE).unwrap(),
            combinator: combine_features,
            empty_attr: Attribution::new(DG_EMPTY_ATTRIBUTION).unwrap(),
            top_cell_attr: Attribution::new(DG_TOP_CELL_ATTRIBUTION).unwrap(),
            top_val_attr: Attribution::new(DG_TOP_VAL_ATTRIBUTION).unwrap(),
            no_vals_attr: Attribution::new(DG_NO_VALS_ATTRIBUTION).unwrap(),
            one_val_attr: Attribution::new(DG_ONE_VAL_ATTRIBUTION).unwrap(),
            no_cells_attr: Attribution::new(DG_NO_CELLS_ATTRIBUTION).unwrap(),
            one_cell_attr: Attribution::new(DG_ONE_CELL_ATTRIBUTION).unwrap(),
        }
    }

    // {NUM_POSSIBLE: -100} I.e., highly prioritize cells with the fewest
    // possible values. Default behavior for combining features in feature
    // vectors is to take the maximum.
    pub fn default() -> Self {
        let mut weights = FeatureVec::new();
        weights.add(&FeatureKey::new(NUM_POSSIBLE_FEATURE).unwrap(), -100.0);
        Self::new(true, weights, |_, a, b| f64::max(a, b))
    }

    // Same as default but with no positive constraint.
    pub fn default_negative() -> Self {
        let mut weights = FeatureVec::new();
        weights.add(&FeatureKey::new(NUM_POSSIBLE_FEATURE).unwrap(), -100.0);
        Self::new(false, weights, |_, a, b| f64::max(a, b))
    }
}

enum SRChoice<V: Value> {
    Cell(Index),
    ValueInRegion(V, Vec<Index>),
}

impl <V: Value, O: Overlay, S: State<V, O>> Ranker<V, O, S> for StdRanker {
    fn top(&self, grid: &DecisionGrid<V>, puzzle: &S) -> BranchPoint<V> {
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
                let score = <Self as Ranker<V, O, S>>::score(self, &mut fv);
                if top_choice.is_none() || score > top_score {
                    top_score = score;
                    top_choice = Some(SRChoice::Cell([r, c]));
                }
            }
        }
        let overlay = puzzle.overlay();
        for dim in 0..overlay.partition_dimension() {
            for p in 0..overlay.n_partitions(dim) {
                if let Some(mut info) = self.region_info(grid, puzzle, dim, p) {
                    for v in V::possibilities() {
                        let uv = v.to_uval();
                        if info.filled.contains(uv) {
                            continue;
                        }
                        let score = <Self as Ranker<V, O, S>>::score(
                            self, info.feature_vecs.get_mut(uv),
                        );
                        if top_choice.is_none() || score > top_score {
                            top_score = score;
                            top_choice = Some(SRChoice::ValueInRegion(v, info.cell_choices.get(uv).clone()));
                        }
                    }
                }
            }
        }
        match top_choice {
            Some(SRChoice::Cell(index)) => {
                BranchPoint::for_cell(0, self.top_cell_attr, index, unpack_values::<V>(&grid.get(index).0))
            },
            Some(SRChoice::ValueInRegion(val, alternatives)) => {
                BranchPoint::for_value(0, self.top_val_attr, val, alternatives)
            },
            None => BranchPoint::empty(0, self.empty_attr),
        }
    }

    fn score(&self, fv: &mut FeatureVec<FVMaybeNormed>) -> f64 {
        fv.normalize_and(self.combinator).dot_product(&self.weights)
    }

    fn to_constraint_result(&self, grid: &DecisionGrid<V>, puzzle: &S) -> ConstraintResult<V> {
        for r in 0..grid.rows() {
            for c in 0..grid.cols() {
                if puzzle.get([r, c]).is_none() {
                    let cell = &grid.get([r, c]).0;
                    if cell.len() == 0 {
                        return ConstraintResult::Contradiction(self.no_vals_attr)
                    } else if cell.len() == 1 {
                        let v = unpack_values::<V>(cell)[0];
                        return ConstraintResult::Certainty(
                            CertainDecision::new([r, c], v),
                            self.one_val_attr,
                        );
                    }
                }
            }
        }
        let overlay = puzzle.overlay();
        for dim in 0..overlay.partition_dimension() {
            for p in 0..overlay.n_partitions(dim) {
                if let Some(info) = self.region_info(grid, puzzle, dim, p) {
                    for v in V::possibilities() {
                        let uv = v.to_uval();
                        if info.filled.contains(uv) {
                            continue;
                        }
                        let choices = info.cell_choices.get(uv);
                        if choices.len() == 0 {
                            return ConstraintResult::Contradiction(self.no_cells_attr);
                        } else if choices.len() == 1 {
                            return ConstraintResult::Certainty(
                                CertainDecision::new(choices[0], v),
                                self.one_cell_attr,
                            );
                        }
                    }
                }
            }
        }
        ConstraintResult::Ok
    }

    fn region_info(
        &self, grid: &DecisionGrid<V>, puzzle: &S, dim: usize, p: usize,
    ) -> Option<RankerRegionInfo<V>> {
        if !self.positive_constraint {
            return None;
        }
        let mut info = RankerRegionInfo::new();
        for index in puzzle.overlay().partition_iter(dim, p) {
            if let Some(val) = puzzle.get(index) {
                let uv = val.to_uval();
                info.filled.insert(uv);
                let cc = info.cell_choices.get_mut(uv);
                cc.clear();
                cc.push(index);
                continue;
            }
            let g = grid.get(index);
            for uv in g.0.iter() {
                info.cell_choices.get_mut(uv).push(index);
                info.feature_vecs.get_mut(uv).extend(&g.1);
            }
        }
        for v in V::possibilities() {
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
        Some(info)
    }
}
