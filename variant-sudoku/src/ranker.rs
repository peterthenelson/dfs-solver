use crate::core::{empty_map, empty_set, readable_feature, unpack_values, Attribution, BranchPoint, CertainDecision, ConstraintResult, DecisionGrid, FVMaybeNormed, FVNormed, FeatureKey, FeatureVec, Index, State, UInt, Value, WithId};
use crate::sudoku::{Overlay, SState, SVal};

/// A ranker finds the "best" place in the grid to make a guess. In theory, we
/// could extend this to return multiple guesses, but since a given index
/// provides a mutually exclusive and exhaustive set of guesses, there isn't
/// really a need.
pub trait Ranker<U: UInt, S: State<U>> {
    // Note: the ranker must not suggest already filled cells.
    fn top(&self, grid: &DecisionGrid<U, S::Value>, puzzle: &S) -> BranchPoint<U, S>;

    // Collapse a DecisionGrid into a ConstraintResult, returning any Certainty
    // or Contradiction that is present. This must be compatible with top() --
    // i.e., top() must always return something possible if no Contradiction is
    // found here.
    fn to_constraint_result(&self, grid: &DecisionGrid<U, S::Value>, puzzle: &S) -> ConstraintResult<U, S::Value>;
}

pub const NUM_POSSIBLE_FEATURE: &str = "NUM_POSSIBLE";
pub const DG_NO_VALS_ATTRIBUTION: &str = "DG_CELL_NO_VALS";
pub const DG_ONE_VAL_ATTRIBUTION: &str = "DG_CELL_ONE_VAL";
pub const DG_NO_CELLS_ATTRIBUTION: &str = "DG_VAL_NO_CELLS";
pub const DG_ONE_CELL_ATTRIBUTION: &str = "DG_VAL_ONE_CELL";

/// A linear scorer. Note that NUM_POSSIBLE is the (most important!) feature
/// that indicates how many possible values are left for a cell.
pub struct LinearRanker {
    weights: FeatureVec<FVNormed>,
    num_possible: FeatureKey<WithId>,
    no_vals_attribution: Attribution<WithId>,
    one_val_attribution: Attribution<WithId>,
}

impl LinearRanker {
    pub fn new(feature_weights: FeatureVec<FVMaybeNormed>) -> Self {
        // Put the dummy feature NUM_POSSIBLE into the registry.
        let mut num_possible = FeatureKey::new(NUM_POSSIBLE_FEATURE);
        let mut weights = feature_weights.clone();
        weights.normalize(|id, _, _| panic!("Duplicate feature in weights vec: {:?}", readable_feature(id)));
        LinearRanker {
            weights: weights.try_normalized().unwrap().clone(),
            num_possible: num_possible.unwrap(),
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

impl <U: UInt, S: State<U>> Ranker<U, S> for LinearRanker {
    fn top(&self, grid: &DecisionGrid<U, S::Value>, puzzle: &S) -> BranchPoint<U, S> {
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
            BranchPoint::for_cell(0, index, unpack_values(&grid.get(index).0))
        } else {
            BranchPoint::empty(0)
        }
    }

    fn to_constraint_result(&self, grid: &DecisionGrid<U, S::Value>, puzzle: &S) -> ConstraintResult<U, S::Value> {
        for r in 0..grid.rows() {
            for c in 0..grid.cols() {
                if puzzle.get([r, c]).is_none() {
                    let cell = &grid.get([r, c]).0;
                    if cell.len() == 0 {
                        return ConstraintResult::Contradiction(self.no_vals_attribution.clone());
                    } else if cell.len() == 1 {
                        let v = unpack_values::<U, S::Value>(cell)[0];
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
    no_vals_attribution: Attribution<WithId>,
    one_val_attribution: Attribution<WithId>,
    no_cells_attribution: Attribution<WithId>,
    one_cell_attribution: Attribution<WithId>,
}

impl OverlaySensitiveLinearRanker {
    pub fn new(feature_weights: FeatureVec<FVMaybeNormed>, combine_features: fn (usize, f64, f64) -> f64) -> Self {
        // Put the dummy feature NUM_POSSIBLE into the registry.
        let mut num_possible = FeatureKey::new(NUM_POSSIBLE_FEATURE);
        let mut weights = feature_weights.clone();
        weights.normalize(|id, _, _| panic!("Duplicate feature in weights vec: {:?}", readable_feature(id)));
        OverlaySensitiveLinearRanker {
            weights: weights.try_normalized().unwrap().clone(),
            num_possible: num_possible.unwrap(),
            combinator: combine_features,
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
}

enum OSLRChoice<const MIN: u8, const MAX: u8> {
    Cell(Index),
    ValueInRegion(SVal<MIN, MAX>, Vec<Index>),
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8, O: Overlay>
Ranker<u8, SState<N, M, MIN, MAX, O>> for OverlaySensitiveLinearRanker {
    fn top(&self, grid: &DecisionGrid<u8, SVal<MIN, MAX>>, puzzle: &SState<N, M, MIN, MAX, O>) -> BranchPoint<u8, SState<N, M, MIN, MAX, O>> {
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
                let mut filled = empty_set::<u8, SVal<MIN, MAX>>();
                let mut alternatives = empty_map::<u8, SVal::<MIN, MAX>, Vec<_>>();
                let mut fvs = empty_map::<u8, SVal::<MIN, MAX>, FeatureVec<FVMaybeNormed>>();
                for index in overlay.partition_iter(dim, p) {
                    if let Some(val) = puzzle.get(index) {
                        filled.insert(val.to_uval());
                        continue;
                    }
                    let g = grid.get(index);
                    for uv in g.0.iter() {
                        alternatives.get_mut(uv).push(index);
                        fvs.get_mut(uv).extend(&g.1);
                    }
                }
                for v in SVal::<MIN, MAX>::possibilities() {
                    let uv = v.to_uval();
                    if filled.contains(uv) {
                        continue;
                    }
                    fvs.get_mut(uv).add(&self.num_possible, alternatives.get(uv).len() as f64);
                    let score = fvs.get_mut(uv).normalize_and(self.combinator).dot_product(&self.weights);
                    if top_choice.is_none() || score > top_score {
                        top_score = score;
                        top_choice = Some(OSLRChoice::ValueInRegion(v, alternatives.get(uv).clone()));
                    }
                }
            }
        }
        match top_choice {
            Some(OSLRChoice::Cell(index)) => {
                BranchPoint::for_cell(0, index, unpack_values(&grid.get(index).0))
            },
            Some(OSLRChoice::ValueInRegion(val, alternatives)) => {
                BranchPoint::for_value(0, val, alternatives)
            },
            None => BranchPoint::empty(0),
        }
    }

    fn to_constraint_result(&self, grid: &DecisionGrid<u8, SVal<MIN, MAX>>, puzzle: &SState<N, M, MIN, MAX, O>) -> ConstraintResult<u8, SVal<MIN, MAX>> {
        for r in 0..grid.rows() {
            for c in 0..grid.cols() {
                if puzzle.get([r, c]).is_none() {
                    let cell = &grid.get([r, c]).0;
                    if cell.len() == 0 {
                        return ConstraintResult::Contradiction(self.no_vals_attribution.clone())
                    } else if cell.len() == 1 {
                        let v = unpack_values::<u8, SVal<MIN, MAX>>(cell)[0];
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
                let mut count = empty_map::<u8, SVal<MIN, MAX>, usize>();
                let mut first_index = empty_map::<u8, SVal<MIN, MAX>, Option<Index>>();
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
