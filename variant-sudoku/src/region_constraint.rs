use std::{collections::{HashMap, HashSet}, fmt::Debug, marker::PhantomData};
use crate::{color_util::color_fib_palette, constraint::Constraint, core::{Attribution, ConstraintResult, Error, Feature, Index, Key, Overlay, RegionLayer, State, Stateful, VBitSet, VSet, VSetMut, Value}, illegal_move::IllegalMove, ranker::RankingInfo};

pub struct Region<V: Value> {
    cells: Vec<Index>,
    _marker: PhantomData<V>,
}

pub struct RegionConstraint<V: Value> {
    layer: Key<RegionLayer>,
    regions: Vec<Region<V>>,
    remaining: Vec<VBitSet<V>>,
    colors: Vec<(u8, u8, u8)>,
    cell_to_region: HashMap<Index, usize>,
    region_feature: Key<Feature>,
    region_conflict_attr: Key<Attribution>,
    illegal: IllegalMove<V>,
}

pub struct RegionContraintBuilder<'a, V: Value, O: Overlay>(&'a mut O, Key<RegionLayer>, HashSet<Index>, PhantomData<V>);

impl <'a, V: Value, O: Overlay> RegionContraintBuilder<'a, V, O> {
    pub fn new(overlay: &'a mut O, layer_name: &'static str) -> Self {
        let layer = Key::register(layer_name);
        overlay.add_region_layer(layer);
        Self(overlay, layer, HashSet::new(), PhantomData)
    }

    pub fn layer(&self) -> Key<RegionLayer> { self.1 }

    pub fn region(&mut self, cells: Vec<Index>) -> Region<V> {
        let mut covered = HashSet::new();
        for c in &cells {
            if covered.contains(c) {
                panic!("Two duplicate cells in region constraint!: {:?}", c);
            }
            covered.insert(*c);
            if self.2.contains(c) {
                panic!("Two regions overlap at cell: {:?}", c);
            }
            self.2.insert(*c);
        }
        if cells.len() > V::cardinality() {
            panic!("Cannot have a region constraint with more cells than \
                    there are possible values! ({} > {})", cells.len(),
                    V::cardinality());
        }
        let positive_constraint = cells.len() == V::cardinality();
        self.0.add_region_in_layer(self.1, positive_constraint, cells.clone());
        Region::<V> { cells, _marker: PhantomData }
    }
}

pub const REGION_CONSTRAINT_FEATURE: &str = "REGION_CONSTRAINT";
pub const REGION_CONSTRAINT_CONFLICT_ATTR: &str = "REGION_CONSTRAINT_CONFLICT";

impl <V: Value> RegionConstraint<V> {
    pub fn new(layer: &'static str, regions: Vec<Region<V>>) -> Self {
        let remaining = vec![VBitSet::<V>::full(); regions.len()];
        let mut cell_to_region = HashMap::new();
        for (i, r) in regions.iter().enumerate() {
            for c in &r.cells {
                cell_to_region.insert(*c, i);
            }
        }
        let colors = color_fib_palette((200, 0, 200), regions.len(), 50.0);
        Self {
            layer: Key::register(layer),
            regions,
            remaining,
            colors,
            cell_to_region,
            region_feature: Key::register(REGION_CONSTRAINT_FEATURE),
            region_conflict_attr: Key::register(REGION_CONSTRAINT_CONFLICT_ATTR),
            illegal: IllegalMove::new(),
        }
    }
}

impl <V: Value> Debug for RegionConstraint<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.illegal.write_dbg(f)?;
        write!(f, "  Layer = {}\n", self.layer.name())?;
        for (i, rem) in self.remaining.iter().enumerate() {
            write!(f, "  [{}] remaining: {}\n", i, rem.to_string())?;
        }
        Ok(())
    }
}

impl <V: Value> Stateful<V> for RegionConstraint<V> {
    fn reset(&mut self) {
        self.remaining = vec![VBitSet::<V>::full(); self.regions.len()];
        self.illegal.reset();
    }

    fn apply(&mut self, index: Index, value: V) -> Result<(), Error> {
        self.illegal.check_unset()?;
        if let Some(i) = self.cell_to_region.get(&index) {
            let rem = &mut self.remaining[*i];
            if rem.contains(&value) {
                rem.remove(&value);
            } else {
                self.illegal.set(index, value, self.region_conflict_attr);
                return Ok(());
            }
        }
        Ok(())
    }

    fn undo(&mut self, index: Index, value: V) -> Result<(), Error> {
        if self.illegal.undo(index, value)? {
            return Ok(());
        }
        if let Some(i) = self.cell_to_region.get(&index) {
            self.remaining[*i].insert(&value);
        }
        Ok(())
    }
}

impl <V: Value, O: Overlay> Constraint<V, O> for RegionConstraint<V> {
    fn name(&self) -> Option<String> { Some("RegionConstraint".to_string()) }

    fn check(&self, puzzle: &State<V, O>, ranking: &mut RankingInfo<V>) -> ConstraintResult<V> {
        let grid = ranking.cells_mut();
        if let Some(c) = self.illegal.to_contradiction() {
            return c;
        }
        for (i, r) in self.regions.iter().enumerate() {
            for c in &r.cells {
                if puzzle.get(*c).is_some() {
                    continue;
                }
                let cell = grid.get_mut(*c);
                cell.0.intersect_with(&self.remaining[i]);
                cell.1.add(&self.region_feature, 1.0);
            }
        }
        ConstraintResult::Ok
    }

    fn debug_at(&self, _: &State<V, O>, index: Index) -> Option<String> {
        let header = "RegionConstraintChecker:\n";
        if let Some(s) = self.illegal.debug_at(index) {
            return Some(format!("{}  {}", header, s));
        }
        if let Some(i) = self.cell_to_region.get(&index) {
            return Some(format!(
                "{}  Unused vals in this region: {}\n",
                header,
                self.remaining[*i].to_string(),
            ))
        }
        None
    }

    fn debug_highlight(&self, _: &State<V, O>, index: Index) -> Option<(u8, u8, u8)> {
        if let Some(c) = self.illegal.debug_highlight(index) {
            return Some(c);
        }
        if let Some(i) = self.cell_to_region.get(&index) {
            return Some(self.colors[*i]);
        }
        None
    }
}

#[cfg(test)]
mod test {
    use crate::{constraint::{test_util::{assert_contradiction, assert_no_contradiction}, MultiConstraint}, ranker::{Ranker, StdRanker}, solver::test_util::PuzzleReplay, sudoku::{four_standard_overlay, FourStdOverlay, FourStdVal, StdChecker}};
    use super::*;

    // This is a 4x4 puzzle with a region in the middle 4 squares. Call with
    // different givens and an expectation for it to return a contradiction (or
    // not).
    fn assert_region_constraint_result(
        setup: &str, 
        expected: Option<&'static str>,
    ) {
        let mut overlay = four_standard_overlay();
        let ranker = StdRanker::default();
        let regions =  {
            let mut rb = RegionContraintBuilder::<FourStdVal, FourStdOverlay>::new(
                &mut overlay, "CUSTOM",
            );
            vec![
                rb.region(vec![[1, 1], [1, 2], [2, 1], [2, 2]]),
            ]
        };
        let mut puzzle = overlay.parse_state::<FourStdVal>(setup).unwrap();
        let mut constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(&overlay),
            RegionConstraint::new("CUSTOM", regions),
        ]);
        let result = PuzzleReplay::new(&mut puzzle, &ranker, &mut constraint, None).replay().unwrap();
        if let Some(attr) = expected {
            assert_contradiction(result, attr);
        } else {
            assert_no_contradiction(result);
        }
    }

    #[test]
    fn test_region_constraint_contradiction() {
        let setup: &str = ". .|. .\n\
                           . 1|. .\n\
                           ---+---\n\
                           . .|1 .\n\
                           . .|. .\n";
        assert_region_constraint_result(setup, Some("REGION_CONSTRAINT_CONFLICT"));
    }

    #[test]
    fn test_region_constraint_valid_fill() {
        let setup: &str = "2 4|1 3\n\
                           3 1|2 4\n\
                           ---+---\n\
                           1 3|4 2\n\
                           4 2|3 1\n";
        assert_region_constraint_result(setup, None);
    }

    #[test]
    fn test_region_constraint_positive() {
        let mut overlay = four_standard_overlay();
        let ranker = StdRanker::default();
        let regions =  {
            let mut rb = RegionContraintBuilder::<FourStdVal, FourStdOverlay>::new(
                &mut overlay, "CUSTOM",
            );
            vec![
                rb.region(vec![[1, 1], [1, 2], [2, 1], [2, 2]]),
            ]
        };
        let setup: &str = ". .|. .\n\
                           . 1|2 .\n\
                           ---+---\n\
                           . 3|. .\n\
                           . .|. .\n";
        let mut puzzle = overlay.parse_state::<FourStdVal>(setup).unwrap();
        let mut constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(&overlay),
            RegionConstraint::new("CUSTOM", regions),
        ]);
        let result = PuzzleReplay::new(&mut puzzle, &ranker, &mut constraint, None).replay().unwrap();
        assert_no_contradiction(result);
        let mut ri = ranker.init_ranking(&puzzle);
        let bp = ranker.rank(5, &mut ri, &puzzle);
        if let ConstraintResult::Certainty(d, a) = bp.0 {
            assert_eq!(d.index, [2, 2]);
            assert_eq!(d.value.val(), 4);
            assert_eq!(a.name(), "DG_VAL_ONE_CELL");
        } else {
            panic!("Expected positive constraint to give a certainty; got {:?}", bp);
        }
    }
}