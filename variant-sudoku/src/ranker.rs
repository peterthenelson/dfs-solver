use std::{fmt::Display, marker::PhantomData};
use crate::core::{readable_key, Attribution, BranchPoint, CertainDecision, ConstraintResult, Error, Feature, Index, Key, Overlay, RegionLayer, State, VBitSet, VDenseMap, VMap, VMapMut, VSet, VSetMut, Value};

/// Marker structs to distinguish scoring-related structures in an indeterminate
/// state vs one where things have definitely been normalized and scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Raw;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scored;

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureVec<S> {
    features: Vec<(usize, f64)>,
    // Whether the features are sorted by id and duplicate ids have been dropped
    // or merged.
    normalized: bool,
    // The score of this vector (against some Scorer).
    score: Option<f64>,
    _marker: PhantomData<S>,
}

impl FeatureVec<Raw> {
    pub fn new() -> Self {
        FeatureVec { features: Vec::new(), normalized: true, score: None, _marker: PhantomData }
    }

    pub fn from_pairs(weights: Vec<(&'static str, f64)>) -> Self {
        let mut fv = FeatureVec::<Raw>::new();
        for (k, v) in weights {
            fv.add(&Key::<Feature>::register(k), v);
        }
        fv
    }

    pub fn add(&mut self, key: &Key<Feature>, value: f64) {
        let id = key.id();
        self.features.push((id, value));
        self.normalized = false;
        self.score = None;
    }

    pub fn extend(&mut self, other: &FeatureVec<Raw>) {
        self.features.extend_from_slice(&other.features);
        self.normalized = false;
        self.score = None;
    }

    pub fn normalize(&mut self, combine: fn(usize, f64, f64) -> f64) {
        if self.normalized {
            return;
        } else if self.features.is_empty() {
            self.normalized = true;
            return;
        }
        self.features.sort_by(|a, b| a.0.cmp(&b.0));
        let mut dst = 0;
        let mut dst_id = self.features[dst].0;
        for src in 1..self.features.len() {
            let src_id = self.features[src].0;
            if src_id == dst_id {
                self.features[dst].1 = combine(src_id, self.features[dst].1, self.features[src].1);
            } else {
                dst += 1;
                self.features[dst] = self.features[src];
                dst_id = src_id;
            }
        }
        self.features.truncate(dst + 1);
        self.normalized = true;
    }

    pub fn score_against<S: Scorer>(&mut self, combine: fn(usize, f64, f64) -> f64, scorer: &S) -> f64 {
        self.normalize(combine);
        let score = scorer.score(&self.features);
        self.score = Some(score);
        score
    }

    pub fn try_scored(&self) -> Result<&FeatureVec<Scored>, Error> {
        if self.score.is_some() {
            Ok(unsafe { std::mem::transmute(self) })
        } else {
            Err(Error::new("FeatureVec is not scored"))
        }
    }

}

impl Default for FeatureVec<Raw> {
    fn default() -> Self {
        FeatureVec::new()
    }
}

impl <S> Display for FeatureVec<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for (i, (k, v)) in self.features.iter().enumerate() {
            if let Some(feature) = readable_key::<Feature>(*k) {
                write!(f, "{} => {}", feature.name(), *v)?;
            } else {
                write!(f, "??? => {}", *v)?;
            }
            if i+1 < self.features.len() {
                write!(f, ", ")?;
            }
        }
        write!(f, "}}")
    }
}

impl FeatureVec<Scored> {
    pub fn get(&self, key: &Key<Feature>) -> Option<f64> {
        let id = key.id();
        for (k, v) in &self.features {
            if *k == id {
                return Some(*v);
            }
        }
        None
    }

    pub fn score(&self) -> f64 {
        self.score.unwrap()
    }

    pub fn decay(&self) -> &FeatureVec<Raw> {
        unsafe { std::mem::transmute(self) }
    }
}

/// A Scorer is a function from (normalized) FeatureVecs to a score.
pub trait Scorer {
    fn score(&self, normalized_features: &Vec<(usize, f64)>) -> f64;
}

/// A FeatureVec<CoVec> is a FeatureVec that acts as a (linear) Scorer against
/// other FeatureVecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoVec;

impl FeatureVec<CoVec> {
    pub fn from_raw(mut weights: FeatureVec<Raw>) -> Self {
        weights.normalize(|id, _, _| panic!("Duplicate feature in weights vec: {:?}", readable_key::<Feature>(id)));
        unsafe { std::mem::transmute(weights) }
    }
}

impl Scorer for FeatureVec<CoVec> {
    fn score(&self, normalized_features: &Vec<(usize, f64)>) -> f64 {
        let mut sum = 0.0;
        let (mut it1, mut it2) = (self.features.iter(), normalized_features.iter());
        let mut i = it1.next();
        let mut j = it2.next();
        while i.is_some() && j.is_some() {
            let (i_id, i_val) = i.unwrap();
            let (j_id, j_val) = j.unwrap();
            if i_id == j_id {
                sum += i_val * j_val;
                i = it1.next();
                j = it2.next();
            } else if i_id < j_id {
                i = it1.next();
            } else {
                j = it2.next();
            }
        }
        sum
    }
}

/// This is a grid of VBitSets and FeatureVecs. It is used to represent the
/// not-yet-ruled-out values for each cell in the grid, along with features
/// attached to each cell.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionGrid<V: Value, S> {
    rows: usize,
    cols: usize,
    grid: Box<[(VBitSet<V>, FeatureVec<Raw>)]>,
    top: Option<(Index, f64)>,
    scored: bool,
    _marker: PhantomData<(V, S)>,
}

impl <V: Value, S> DecisionGrid<V, S> {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, index: Index) -> (&VBitSet<V>, &FeatureVec<Raw>) {
        let (s, f) = &self.grid[index[0] * self.cols + index[1]];
        (s, f)
    }

    // Note: This may still be None when scored is true (i.e., if the puzzle is
    // full).
    pub fn top(&self) -> Option<(Index, f64)> {
        self.top
    }
}

impl<V: Value> DecisionGrid<V, Raw> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            grid: vec![(VBitSet::<V>::empty(), FeatureVec::new()); rows * cols].into_boxed_slice(),
            top: None,
            scored: false,
            _marker: PhantomData,
        }
    }

    pub fn full(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            grid: vec![(VBitSet::<V>::full(), FeatureVec::new()); rows * cols].into_boxed_slice(),
            top: None,
            scored: false,
            _marker: PhantomData,
        }
    }

    pub fn get_mut(&mut self, index: Index) -> (&mut VBitSet<V>, &mut FeatureVec<Raw>) {
        let (s, f) = &mut self.grid[index[0] * self.cols + index[1]];
        self.top = None;
        (s, f)
    }

    /// Intersects the possible values of this grid with the possible values of
    /// another grid. Also merges the features of the two grids.
    pub fn combine_with(&mut self, other: &DecisionGrid<V, Raw>) {
        assert!(self.rows == other.rows && self.cols == other.cols,
                "Cannot combine grids of different sizes");
        self.top = None;
        for i in 0..self.rows {
            for j in 0..self.cols {
                let a = self.get_mut([i, j]);
                let b = other.get([i, j]);
                a.0.intersect_with(b.0);
                a.1.extend(b.1);
            }
        }
    }

    pub fn score_against<O: Overlay, S: Scorer>(&mut self, combine: fn(usize, f64, f64) -> f64, puzzle: &State<V, O>, scorer: &S) -> Option<(Index, f64)> {
        self.top = None;
        for r in 0..self.rows {
            for c in 0..self.cols {
                if puzzle.get([r, c]).is_some() {
                    continue;
                }
                let cell = &mut self.grid[r * self.cols + c];
                let score = cell.1.score_against(combine, scorer);
                self.top = self.top.map_or(Some(([r, c], score)), |(i, s)| {
                    Some(if score > s { ([r, c], score) } else { (i, s) })
                });
            }
        }
        self.scored = true;
        self.top
    }

    pub fn try_scored(&self) -> Result<&DecisionGrid<V, Scored>, Error> {
        if self.scored {
            Ok(unsafe { std::mem::transmute(self) })
        } else {
            Err(Error::new("DecisionGrid is not scored"))
        }
    }
}

impl<V: Value> DecisionGrid<V, Scored> {
    pub fn scored_feature(&self, index: Index) -> &FeatureVec<Scored> {
        let (_, f)= &self.grid[index[0] * self.cols + index[1]];
        f.try_scored().unwrap()
    }
}

/// Constraints help the Ranker make decisions by providing information via
/// RankingInfo. The available grids should be consistent with the regions
/// visible in the Overlay.
#[derive(Debug, Clone)]
pub struct RankingInfo<V: Value> {
    // The primary way to provide ranking-relevant information is to add
    // features to or restrict available values in a DecisionGrid.
    cells: DecisionGrid<V, Raw>,

    // The feature for # of possibilities in the value-in-region FeatureVecs.
    val_possible: Key<Feature>,

    // The max score found, if any (only applicable after rank() call).
    // TODO: Keep track of min score too.
    top_score: Option<f64>,

    // TODO: What value-in-region- or region-centric information can Constraints
    // provide to the ranker? If any, put it here.
}

impl <V: Value> RankingInfo<V> {
    pub fn new(cells: DecisionGrid<V, Raw>) -> Self {
        Self {
            cells,
            val_possible: Key::register(DG_VAL_POSSIBLE_FEATURE),
            top_score: None,
        }
    }

    pub fn cells(&self) -> &DecisionGrid<V, Raw> {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut DecisionGrid<V, Raw> {
        &mut self.cells
    }

    pub fn score_against<O: Overlay, S: Scorer>(
        &mut self, combine: fn(usize, f64, f64) -> f64,
        puzzle: &State<V, O>, scorer: &S,
    ) -> Option<(Index, f64)> {
        self.cells.score_against(combine, puzzle, scorer)
    }

    /// Give a value-in-region-centric view of the grid (i.e., the positive
    /// positive constraints implied by the overlay).
    pub fn region_info<O: Overlay>(
        &self, puzzle: &State<V, O>, layer: Key<RegionLayer>, p: usize,
    ) -> RankerRegionInfo<V> {
        let mut info = RankerRegionInfo::new();
        for index in puzzle.overlay().region_iter(layer, p) {
            if let Some(val) = puzzle.get(index) {
                info.filled.insert(&val);
                let cc = info.cell_choices.get_mut(&val);
                cc.clear();
                cc.push(index);
                continue;
            }
            let g = self.cells.get(index);
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
        info
    }

    pub fn top_score(&self) -> Option<f64> { self.top_score }
}

/// A ranker finds the "best" place in the grid to make a guess. This could
/// either be a cell ("Here are the mutually exclusive and exhaustive
/// possible values that go in [0, 0]") or a value in a region ("here are the
/// mutually exclusive and exhaustive possible cells where the 9 in row 3 can
/// go"). In theory, the ranker can choose to order the values/cells in the
/// branch point, but the StdRanker does not do so, and the RankingInfo doesn't
/// currently provide useful guidance to base such a decision on.
pub trait Ranker<V: Value, O: Overlay> {
    fn init_ranking(&self, puzzle: &State<V, O>) -> RankingInfo<V>;

    // Note: the ranker must not suggest already filled cells.
    fn rank(&self, step: usize, ranking: &mut RankingInfo<V>, puzzle: &State<V, O>) -> (ConstraintResult<V>, Option<BranchPoint<V>>);

    // If early exit prevented ranking from completing, finish computing scores.
    fn ensure_scored(&self, ranking: &mut RankingInfo<V>, puzzle: &State<V, O>);

    // Score the region info.
    fn score_region_info(&self, region_info: &mut RankerRegionInfo<V>) -> Option<(V, Vec<Index>, f64)> ;
}

pub struct RankerRegionInfo<V: Value> {
    // Values that have already been filled into the puzzle.
    pub filled: VBitSet<V>,
    // Cells that a given value can go into.
    pub cell_choices: VDenseMap<V, Vec<Index>>,
    // Feature vectors for a given value.
    pub feature_vecs: VDenseMap<V, FeatureVec<Raw>>,
    _marker: PhantomData<V>,
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
pub struct StdRanker<O: Overlay> {
    cell_weights: FeatureVec<CoVec>,
    val_weights: FeatureVec<CoVec>,
    cell_possible: Key<Feature>,
    combinator: fn (usize, f64, f64) -> f64,
    empty_attr: Key<Attribution>,
    top_cell_attr: Key<Attribution>,
    top_val_attr: Key<Attribution>,
    no_vals_attr: Key<Attribution>,
    one_val_attr: Key<Attribution>,
    no_cells_attr: Key<Attribution>,
    one_cell_attr: Key<Attribution>,
    _marker: PhantomData<O>,
}

impl <V: Value> RankerRegionInfo<V> {
    pub fn new() -> Self {
        Self {
            filled: VBitSet::<V>::empty(),
            cell_choices: VDenseMap::<V, Vec<Index>>::empty(),
            feature_vecs: VDenseMap::<V, FeatureVec<Raw>>::empty(),
            _marker: PhantomData,
        }
    }
}

impl <O: Overlay> StdRanker<O> {
    pub fn new(cell_weights: FeatureVec<CoVec>, val_weights: FeatureVec<CoVec>, combine_features: fn (usize, f64, f64) -> f64) -> Self {
        StdRanker {
            cell_weights,
            val_weights,
            cell_possible: Key::register(DG_CELL_POSSIBLE_FEATURE),
            combinator: combine_features,
            empty_attr: Key::register(DG_EMPTY_ATTRIBUTION),
            top_cell_attr: Key::register(DG_TOP_CELL_ATTRIBUTION),
            top_val_attr: Key::register(DG_TOP_VAL_ATTRIBUTION),
            no_vals_attr: Key::register(DG_NO_VALS_ATTRIBUTION),
            one_val_attr: Key::register(DG_ONE_VAL_ATTRIBUTION),
            no_cells_attr: Key::register(DG_NO_CELLS_ATTRIBUTION),
            one_cell_attr: Key::register(DG_ONE_CELL_ATTRIBUTION),
            _marker: PhantomData,
        }
    }

    // Like the default but extended with additional weights.
    pub fn with_additional_weights(weights: FeatureVec<Raw>) -> Self {
        let mut cell_weights = FeatureVec::new();
        cell_weights.add(&Key::register(DG_CELL_POSSIBLE_FEATURE), -10.0);
        cell_weights.extend(&weights);
        let mut val_weights = FeatureVec::new();
        val_weights.add(&Key::register(DG_VAL_POSSIBLE_FEATURE), -10.0);
        val_weights.extend(&weights);
        Self::new(
            FeatureVec::from_raw(cell_weights),
            FeatureVec::from_raw(val_weights),
            |_, a, b| f64::max(a, b),
        )
    }

    // {DB_CELL_POSSIBLE: -10, DB_VAL_POSSIBLE: -10} I.e., highly prioritize
    // cells/values with the fewest possible values/cells. Default behavior for
    // combining features in feature vectors is to take the maximum.
    pub fn default() -> Self {
        let mut cell_weights = FeatureVec::new();
        cell_weights.add(&Key::register(DG_CELL_POSSIBLE_FEATURE), -10.0);
        let mut val_weights = FeatureVec::new();
        val_weights.add(&Key::register(DG_VAL_POSSIBLE_FEATURE), -10.0);
        Self::new(
            FeatureVec::from_raw(cell_weights),
            FeatureVec::from_raw(val_weights),
            |_, a, b| f64::max(a, b),
        )
    }

    fn annotate<V: Value>(&self, ranking: &mut RankingInfo<V>, puzzle: &State<V, O>) {
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

    fn to_constraint_result<V: Value>(&self, ranking: &RankingInfo<V>, puzzle: &State<V, O>) -> ConstraintResult<V> {
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
                let info = ranking.region_info(puzzle, layer, p);
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
        ConstraintResult::Ok
    }
}

enum SRChoice<V: Value> {
    Cell(Index),
    ValueInRegion(V, Vec<Index>),
}

impl <V: Value, O: Overlay> Ranker<V, O> for StdRanker<O> {
    fn init_ranking(&self, puzzle: &State<V, O>) -> RankingInfo<V> {
        let (n, m) = puzzle.overlay().grid_dims();
        let mut cells = DecisionGrid::<V, Raw>::full(n, m);
        for r in 0..n {
            for c in 0..m {
                if let Some(v) = puzzle.get([r, c]) {
                    *cells.get_mut([r, c]).0 = VBitSet::<V>::singleton(&v);
                }
            }
        }
        RankingInfo::new(cells)
    }

    fn rank(&self, step: usize, ranking: &mut RankingInfo<V>, puzzle: &State<V, O>) -> (ConstraintResult<V>, Option<BranchPoint<V>>) {
        self.annotate(ranking, puzzle);
        let top_cell = ranking.score_against(self.combinator, puzzle, &self.cell_weights);
        // TODO: Should be able to score and check for early exits in one pass,
        // avoiding duplicate creation of region_infos
        let cr = self.to_constraint_result(&ranking, puzzle);
        match &cr {
            ConstraintResult::Contradiction(_) | ConstraintResult::Certainty(_, _) => {
                return (cr, None);
            },
            ConstraintResult::Ok => {},
        };
        let mut top_choice = if let Some((i, s)) = top_cell {
            Some((SRChoice::Cell(i), s))
        } else {
            return (ConstraintResult::Ok, Some(BranchPoint::empty(step, self.empty_attr)));
        };
        let overlay = puzzle.overlay();
        for layer in overlay.region_layers() {
            for p in 0..overlay.regions_in_layer(layer) {
                let mut info = ranking.region_info(puzzle, layer, p);
                let top_val = self.score_region_info(&mut info);
                top_choice = top_choice.map(|(ch, s_old)| {
                    if let Some((v, c, s)) = top_val {
                        if s > s_old {
                            return (SRChoice::ValueInRegion(v, c), s);
                        }
                    }
                    (ch, s_old)
                });
            }
        }
        ranking.top_score = Some(top_choice.as_ref().unwrap().1);
        let bp = match top_choice {
            Some((SRChoice::Cell(index), _)) => {
                BranchPoint::for_cell(
                    step, self.top_cell_attr, index,
                    ranking.cells().get(index).0.iter().collect::<Vec<_>>(),
                )
            },
            Some((SRChoice::ValueInRegion(val, alternatives), _)) => {
                BranchPoint::for_value(step, self.top_val_attr, val, alternatives)
            },
            _ => panic!("Should be unreachable!"),
        };
        (ConstraintResult::Ok, Some(bp))
    }

    fn ensure_scored(&self, ranking: &mut RankingInfo<V>, puzzle: &State<V, O>) {
        if ranking.top_score().is_some() {
            return;
        }
        let mut top_score = if let Ok(scored) = ranking.cells_mut().try_scored() {
            scored.top()
        } else {
            ranking.score_against(self.combinator, puzzle, &self.cell_weights)
        }.map(|(_, s)| s);
        if top_score.is_none() {
            return;
        }
        let overlay = puzzle.overlay();
        for layer in overlay.region_layers() {
            for p in 0..overlay.regions_in_layer(layer) {
                let mut info = ranking.region_info(puzzle, layer, p);
                let top_val = self.score_region_info(&mut info);
                top_score = top_score.map(|s_old| {
                    if let Some((_, _, s)) = top_val {
                        if s > s_old {
                            return s
                        }
                    }
                    s_old
                });
            }
        }
        ranking.top_score = top_score;
        
    }

    fn score_region_info(&self, region_info: &mut RankerRegionInfo<V>) -> Option<(V, Vec<Index>, f64)> {
        let mut top_choice = None;
        for v in V::possibilities() {
            if region_info.filled.contains(&v) {
                continue;
            }
            let fv = region_info.feature_vecs.get_mut(&v);
            fv.score_against(self.combinator, &self.val_weights);
            let score = fv.try_scored().unwrap().score();
            top_choice = top_choice.map_or_else(
                || Some((v, region_info.cell_choices.get(&v).clone(), score)).clone(),
                 |(v_old, c_old, s_old)| {
                    Some(if score > s_old { 
                        (v, region_info.cell_choices.get(&v).clone(), score)
                    } else {
                        (v_old, c_old, s_old)
                    })
                });
        }
        top_choice
    }
}
