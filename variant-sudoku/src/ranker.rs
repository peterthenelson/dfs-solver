use std::marker::PhantomData;
use crate::core::{readable_key, Attribution, BranchPoint, CertainDecision, ConstraintResult, DecisionGrid, FVMaybeNormed, FVNormed, Feature, FeatureVec, Index, Key, Overlay, RankingInfo, RegionLayer, Scored, State, Unscored, VBitSet, VDenseMap, VMap, VMapMut, VSet, Value, WithId};

/// A ranker finds the "best" place in the grid to make a guess. This could
/// either be a cell ("Here are the mutually exclusive and exhaustive
/// possible values that go in [0, 0]") or a value in a region ("here are the
/// mutually exclusive and exhaustive possible cells where the 9 in row 3 can
/// go"). In theory, the ranker can choose to order the values/cells in the
/// branch point, but the StdRanker does not do so, and the RankingInfo doesn't
/// currently provide useful guidance to base such a decision on.
pub trait Ranker<V: Value, O: Overlay> {
    fn init_ranking(&self, puzzle: &State<V, O>) -> RankingInfo<V, Unscored>;

    // Note: the ranker must not suggest already filled cells.
    fn rank(&self, step: usize, ranking: RankingInfo<V, Unscored>, puzzle: &State<V, O>) -> (RankingInfo<V, Scored>, ConstraintResult<V>, Option<BranchPoint<V>>);

    // Score for a particular feature vec for a cell. Exposed for debugging reasons.
    fn cell_score(&self, fv: &mut FeatureVec<FVMaybeNormed>) -> f64;

    // Score for a particular feature vec for a val. Exposed for debugging reasons.
    fn val_score(&self, fv: &mut FeatureVec<FVMaybeNormed>) -> f64;

    /// Exposes how the ranker looks at the grid when calculating candidates
    /// in regions (i.e., those implied by positive_constraint). Useful for
    /// debugging without needing to duplicate the implementation of the ranker.
    /// Implementations may return None if they don't generate candidates in
    /// this way.
    fn region_info<S: DGGetter<V>>(
        &self, grid: &DecisionGrid<V, S>, puzzle: &State<V, O>, layer: Key<RegionLayer, WithId>, p: usize,
    ) -> Option<RankerRegionInfo<V>>;
}

// TODO: Remove this once region_info is all refactored away.
pub trait DGGetter<V: Value> where Self: Sized {
    fn get_possible_and_features<'a>(grid: &'a DecisionGrid<V, Self>, index: Index) -> (&'a VBitSet<V>, &'a FeatureVec<FVMaybeNormed>);
}

impl <V: Value> DGGetter<V> for Unscored {
    fn get_possible_and_features<'a>(grid: &'a DecisionGrid<V, Unscored>, index: Index) -> (&'a VBitSet<V>, &'a FeatureVec<FVMaybeNormed>) {
        grid.get(index)
    }
}

impl <V: Value> DGGetter<V> for Scored {
    fn get_possible_and_features<'a>(grid: &'a DecisionGrid<V, Scored>, index: Index) -> (&'a VBitSet<V>, &'a FeatureVec<FVMaybeNormed>) {
        let (p, f, _) = grid.get(index);
        (p, f.decay())
    }
}


pub struct RankerRegionInfo<V: Value> {
    // Values that have already been filled into the puzzle.
    pub filled: VBitSet<V>,
    // Cells that a given value can go into.
    pub cell_choices: VDenseMap<V, Vec<Index>>,
    // Feature vectors for a given value.
    pub feature_vecs: VDenseMap<V, FeatureVec<FVMaybeNormed>>,
    p_v_: PhantomData<V>,
}

pub const DG_CELL_POSSIBLE_FEATURE: &str = "DG_CELL_POSSIBLE";
pub const DG_VAL_POSSIBLE_FEATURE: &str = "DG_VAL_POSSIBLE";
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
    cell_weights: FeatureVec<FVNormed>,
    val_weights: FeatureVec<FVNormed>,
    cell_possible: Key<Feature, WithId>,
    val_possible: Key<Feature, WithId>,
    combinator: fn (usize, f64, f64) -> f64,
    empty_attr: Key<Attribution, WithId>,
    top_cell_attr: Key<Attribution, WithId>,
    top_val_attr: Key<Attribution, WithId>,
    no_vals_attr: Key<Attribution, WithId>,
    one_val_attr: Key<Attribution, WithId>,
    no_cells_attr: Key<Attribution, WithId>,
    one_cell_attr: Key<Attribution, WithId>,
}

impl <V: Value> RankerRegionInfo<V> {
    pub fn new() -> Self {
        Self {
            filled: VBitSet::<V>::empty(),
            cell_choices: VDenseMap::<V, Vec<Index>>::empty(),
            feature_vecs: VDenseMap::<V, FeatureVec<FVMaybeNormed>>::empty(),
            p_v_: PhantomData,
        }
    }
}

impl StdRanker {
    pub fn new(positive_constraint: bool, cell_weights: FeatureVec<FVMaybeNormed>, val_weights: FeatureVec<FVMaybeNormed>, combine_features: fn (usize, f64, f64) -> f64) -> Self {
        let mut cell_weights = cell_weights.clone();
        cell_weights.normalize(|id, _, _| panic!("Duplicate feature in weights vec: {:?}", readable_key::<Feature>(id)));
        let mut val_weights = val_weights.clone();
        val_weights.normalize(|id, _, _| panic!("Duplicate feature in weights vec: {:?}", readable_key::<Feature>(id)));
        StdRanker {
            positive_constraint,
            cell_weights: cell_weights.try_normalized().unwrap().clone(),
            val_weights: val_weights.try_normalized().unwrap().clone(),
            cell_possible: Key::new(DG_CELL_POSSIBLE_FEATURE).unwrap(),
            val_possible: Key::new(DG_VAL_POSSIBLE_FEATURE).unwrap(),
            combinator: combine_features,
            empty_attr: Key::new(DG_EMPTY_ATTRIBUTION).unwrap(),
            top_cell_attr: Key::new(DG_TOP_CELL_ATTRIBUTION).unwrap(),
            top_val_attr: Key::new(DG_TOP_VAL_ATTRIBUTION).unwrap(),
            no_vals_attr: Key::new(DG_NO_VALS_ATTRIBUTION).unwrap(),
            one_val_attr: Key::new(DG_ONE_VAL_ATTRIBUTION).unwrap(),
            no_cells_attr: Key::new(DG_NO_CELLS_ATTRIBUTION).unwrap(),
            one_cell_attr: Key::new(DG_ONE_CELL_ATTRIBUTION).unwrap(),
        }
    }

    // Like the default but extended with additional weights.
    pub fn with_additional_weights(weights: FeatureVec<FVMaybeNormed>) -> Self {
        let mut cell_weights = FeatureVec::new();
        cell_weights.add(&Key::new(DG_CELL_POSSIBLE_FEATURE).unwrap(), -10.0);
        cell_weights.extend(&weights);
        let mut val_weights = FeatureVec::new();
        val_weights.add(&Key::new(DG_VAL_POSSIBLE_FEATURE).unwrap(), -10.0);
        val_weights.extend(&weights);
        Self::new(true, cell_weights, val_weights, |_, a, b| f64::max(a, b))
    }

    // {DB_CELL_POSSIBLE: -10, DB_VAL_POSSIBLE: -10} I.e., highly prioritize
    // cells/values with the fewest possible values/cells. Default behavior for
    // combining features in feature vectors is to take the maximum.
    pub fn default() -> Self {
        let mut cell_weights = FeatureVec::new();
        cell_weights.add(&Key::new(DG_CELL_POSSIBLE_FEATURE).unwrap(), -10.0);
        let mut val_weights = FeatureVec::new();
        val_weights.add(&Key::new(DG_VAL_POSSIBLE_FEATURE).unwrap(), -10.0);
        Self::new(true, cell_weights, val_weights, |_, a, b| f64::max(a, b))
    }

    // Same as default but with no positive constraint.
    pub fn default_negative() -> Self {
        let mut cell_weights = FeatureVec::new();
        cell_weights.add(&Key::new(DG_CELL_POSSIBLE_FEATURE).unwrap(), -10.0);
        let mut val_weights = FeatureVec::new();
        val_weights.add(&Key::new(DG_VAL_POSSIBLE_FEATURE).unwrap(), -10.0);
        Self::new(false, cell_weights, val_weights, |_, a, b| f64::max(a, b))
    }

    fn annotate<V: Value, O: Overlay>(&self, ranking: &mut RankingInfo<V, Unscored>, puzzle: &State<V, O>) {
        let grid = ranking.cells_mut();
        for r in 0..grid.rows() {
            for c in 0..grid.cols() {
                if puzzle.get([r, c]).is_some() {
                    continue;
                }
                let g = grid.get_mut([r, c]);
                let fv = g.1;
                fv.add(&self.cell_possible, g.0.len() as f64);
            }
        }
    }

    fn to_constraint_result<V: Value, O: Overlay>(&self, ranking: &RankingInfo<V, Scored>, puzzle: &State<V, O>) -> ConstraintResult<V> {
        let grid = ranking.cells();
        for r in 0..grid.rows() {
            for c in 0..grid.cols() {
                if puzzle.get([r, c]).is_none() {
                    let cell = &grid.get([r, c]).0;
                    if cell.len() == 0 {
                        return ConstraintResult::Contradiction(self.no_vals_attr)
                    } else if let Some(v) = cell.as_singleton() {
                        return ConstraintResult::Certainty(
                            CertainDecision::new([r, c], v),
                            self.one_val_attr,
                        );
                    }
                }
            }
        }
        let overlay = puzzle.overlay();
        for layer in overlay.region_layers() {
            for p in 0..overlay.regions_in_layer(layer) {
                if let Some(info) = self.region_info(grid, puzzle, layer, p) {
                    for v in V::possibilities() {
                        if info.filled.contains(&v) {
                            continue;
                        }
                        let choices = info.cell_choices.get(&v);
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
}

enum SRChoice<V: Value> {
    Cell(Index),
    ValueInRegion(V, Vec<Index>),
}

impl <V: Value, O: Overlay> Ranker<V, O> for StdRanker {
    fn init_ranking(&self, puzzle: &State<V, O>) -> RankingInfo<V, Unscored> {
        let mut cells = puzzle.overlay().full_decision_grid();
        let (n, m) = puzzle.overlay().grid_dims();
        for r in 0..n {
            for c in 0..m {
                if let Some(v) = puzzle.get([r, c]) {
                    *cells.get_mut([r, c]).0 = VBitSet::<V>::singleton(&v);
                }
            }
        }
        RankingInfo::new(cells)
    }

    fn rank(&self, step: usize, ranking: RankingInfo<V, Unscored>, puzzle: &State<V, O>) -> (RankingInfo<V, Scored>, ConstraintResult<V>, Option<BranchPoint<V>>) {
        let mut ranking = ranking;
        self.annotate(&mut ranking, puzzle);
        let ranking = ranking.into_scored(|fv| {
            <Self as Ranker<V, O>>::cell_score(self, fv)
        });
        let cr = self.to_constraint_result(&ranking, puzzle);
        match &cr {
            ConstraintResult::Contradiction(_) | ConstraintResult::Certainty(_, _) => {
                return (ranking, cr, None);
            },
            ConstraintResult::Ok => {},
        };
        let mut top_choice = None;
        let mut top_score: f64 = f64::MIN;
        let grid = ranking.cells();
        for r in 0..grid.rows() {
            for c in 0..grid.cols() {
                if puzzle.get([r, c]).is_some() {
                    continue;
                }
                let score = grid.get([r, c]).2;
                if score > top_score {
                    top_score = score;
                    top_choice = Some(SRChoice::Cell([r, c]));
                }
            }
        }
        if top_choice.is_none() {
            return (ranking, ConstraintResult::Ok, Some(BranchPoint::empty(step, self.empty_attr)));
        }
        // TODO: Factor this into the RankingInfo and the init_ranking method
        let overlay = puzzle.overlay();
        for layer in overlay.region_layers() {
            for p in 0..overlay.regions_in_layer(layer) {
                if let Some(mut info) = self.region_info(grid, puzzle, layer, p) {
                    for v in V::possibilities() {
                        if info.filled.contains(&v) {
                            continue;
                        }
                        let score = <Self as Ranker<V, O>>::val_score(
                            self,
                            info.feature_vecs.get_mut(&v),
                        );
                        if top_choice.is_none() || score > top_score {
                            top_score = score;
                            top_choice = Some(SRChoice::ValueInRegion(v, info.cell_choices.get(&v).clone()));
                        }
                    }
                }
            }
        }
        let bp = match top_choice {
            Some(SRChoice::Cell(index)) => {
                BranchPoint::for_cell(
                    step, self.top_cell_attr, index,
                    grid.get(index).0.iter().collect::<Vec<_>>(),
                )
            },
            Some(SRChoice::ValueInRegion(val, alternatives)) => {
                BranchPoint::for_value(step, self.top_val_attr, val, alternatives)
            },
            _ => panic!("Should be unreachable!"),
        };
        (ranking, ConstraintResult::Ok, Some(bp))
    }

    fn cell_score(&self, fv: &mut FeatureVec<FVMaybeNormed>) -> f64 {
        fv.normalize_and(self.combinator).dot_product(&self.cell_weights)
    }

    fn val_score(&self, fv: &mut FeatureVec<FVMaybeNormed>) -> f64 {
        fv.normalize_and(self.combinator).dot_product(&self.val_weights)
    }

    fn region_info<S: DGGetter<V>>(
        &self, grid: &DecisionGrid<V, S>, puzzle: &State<V, O>, layer: Key<RegionLayer, WithId>, p: usize,
    ) -> Option<RankerRegionInfo<V>> {
        if !self.positive_constraint {
            return None;
        }
        let mut info = RankerRegionInfo::new();
        for index in puzzle.overlay().region_iter(layer, p) {
            if let Some(val) = puzzle.get(index) {
                info.filled.insert(&val);
                let cc = info.cell_choices.get_mut(&val);
                cc.clear();
                cc.push(index);
                continue;
            }
            let g = S::get_possible_and_features(grid, index);
            for v in g.0.iter() {
                info.cell_choices.get_mut(&v).push(index);
                info.feature_vecs.get_mut(&v).extend(&g.1);
            }
        }
        for v in V::possibilities() {
            if info.filled.contains(&v) {
                continue;
            }
            // TODO: Should we normalize the other feature values by
            // 1/alternatives? Otherwise we're implicitly overweighting
            // towards choosing a ::ValueInRegion over a ::Cell.
            info.feature_vecs.get_mut(&v).add(
                &self.val_possible,
                info.cell_choices.get(&v).len() as f64,
            );
        }
        Some(info)
    }
}
