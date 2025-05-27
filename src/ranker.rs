use crate::core::{DecisionGrid, FKWithId, FVMaybeNormed, FVNormed, FeatureKey, FeatureVec, Index, State, UInt};

/// A ranker finds the "best" place in the grid to make a guess. In theory, we
/// could extend this to return multiple guesses, but since a given index
/// provides a mutually exclusive and exhaustive set of guesses, there isn't
/// really a need.
pub trait Ranker<U: UInt, S: State<U>> {
    // Note: the ranker must not suggest already filled cells.
    fn top(&self, grid: &DecisionGrid<U, S::Value>, puzzle: &S) -> Option<Index>;
}

/// A linear scorer. Note that NUM_POSSIBLE is the (most important!) feature
/// that indicates how many possible values are left for a cell.
pub struct LinearRanker {
    weights: FeatureVec<FVNormed>,
    num_possible: FeatureKey<FKWithId>,
}

pub const NUM_POSSIBLE_FEATURE: &str = "NUM_POSSIBLE";

impl LinearRanker {
    pub fn new(feature_weights: FeatureVec<FVMaybeNormed>) -> Self {
        // Put the dummy feature NUM_POSSIBLE into the registry.
        let mut num_possible = FeatureKey::new(NUM_POSSIBLE_FEATURE);
        let mut weights = feature_weights.clone();
        weights.normalize(|_, _| panic!("Duplicate features in weights vec"));
        LinearRanker {
            weights: weights.try_normalized().unwrap().clone(),
            num_possible: num_possible.unwrap(),
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
    fn top(&self, grid: &DecisionGrid<U, S::Value>, puzzle: &S) -> Option<Index> {
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
                let score = fv.normalize_and(|a, b| a+b).dot_product(&self.weights);
                if top_index.is_none() || score > top_score {
                    top_score = score;
                    top_index = Some([r, c]);
                }
            }
        }
        return top_index;
    }
}
