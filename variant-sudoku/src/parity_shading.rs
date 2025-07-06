use std::{collections::{HashMap, HashSet}, fmt::Debug};
use crate::{constraint::Constraint, core::{Attribution, ConstraintResult, Error, Feature, Index, Key, Overlay, State, Stateful, VBitSet, VSet, VSetMut, Value}, ranker::RankingInfo, sudoku::StdVal};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Parity { Even, Odd }

#[derive(Clone)]
pub struct ParityShading(Index, Parity);

impl ParityShading {
    pub fn valid<const MIN: u8, const MAX: u8>(&self, v: &StdVal<MIN, MAX>) -> bool {
        v.val() % 2 == match self.1 { Parity::Even => 0, Parity::Odd => 1 }
    }

    pub fn values<const MIN: u8, const MAX: u8>(&self) -> VBitSet<StdVal<MIN, MAX>> {
        let mut vs = VBitSet::empty();
        for v in StdVal::<MIN, MAX>::possibilities() {
            if self.valid(&v) {
                vs.insert(&v);
            }
        }
        vs
    }

    pub fn color(&self) -> (u8, u8, u8) {
        match self.1 { Parity::Even => (0, 0, 200), Parity::Odd => (200, 200, 0) }
    }
}

pub struct ParityShadingBuilder;
impl ParityShadingBuilder {
    pub fn new() -> Self { Self {} }

    pub fn even(&self, index: Index) -> ParityShading {
        ParityShading(index, Parity::Even)
    }

    pub fn odd(&self, index: Index) -> ParityShading {
        ParityShading(index, Parity::Odd)
    }
}

pub const PARITY_SHADING_ILLEGAL_ACTION: Error = Error::new_const("A parity-shading violation already exists; can't apply further actions.");
pub const PARITY_SHADING_UNDO_MISMATCH: Error = Error::new_const("Undo value mismatch");
pub const PARITY_SHADING_FEATURE: &str = "PARITY_SHADING";
pub const PARITY_SHADING_CONFLICT_ATTR: &str = "PARITY_SHADING_CONFLICT";

pub struct ParityShadingChecker<const MIN: u8, const MAX: u8> {
    remaining: HashMap<Index, VBitSet<StdVal<MIN, MAX>>>,
    remaining_init: HashMap<Index, VBitSet<StdVal<MIN, MAX>>>,
    colors: HashMap<Index, (u8, u8, u8)>,
    shading_feature: Key<Feature>,
    shading_conflict_attr: Key<Attribution>,
    illegal: Option<(Index, StdVal<MIN, MAX>, Key<Attribution>)>,
}

impl <const MIN: u8, const MAX: u8> ParityShadingChecker<MIN, MAX> {
    pub fn new(shadings: Vec<ParityShading>) -> Self {
        let mut covered = HashSet::new();
        for s in &shadings {
            if covered.contains(&s.0) {
                panic!("Cell {:?} covered by multiple parity shadings!", s.0);
            }
            covered.insert(s.0);
        }
        let remaining_init = HashMap::from_iter(shadings
            .iter()
            .map(|s| (s.0, s.values())),
        );
        let colors = HashMap::from_iter(shadings
            .into_iter()
            .map(|s| (s.0, s.color())),
        );
        Self {
            remaining: remaining_init.clone(),
            remaining_init,
            colors,
            shading_feature: Key::register(PARITY_SHADING_FEATURE),
            shading_conflict_attr: Key::register(PARITY_SHADING_CONFLICT_ATTR),
            illegal: None
        }
    }
}

impl <const MIN: u8, const MAX: u8> Debug for ParityShadingChecker<MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((i, v, a)) = &self.illegal {
            write!(f, "Illegal move: {:?}={:?} ({})\n", i, v, a.name())?;
        }
        for (i, r) in self.remaining.iter() {
            write!(f, " [{:?}] => {}\n", i, r.to_string())?;
        }
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8>
Stateful<StdVal<MIN, MAX>> for ParityShadingChecker<MIN, MAX> {
    fn reset(&mut self) {
        self.remaining = self.remaining_init.clone();
    }

    fn apply(&mut self, index: Index, value: StdVal<MIN, MAX>) -> Result<(), Error> {
        // In theory we could be allow multiple illegal moves and just
        // invalidate and recalculate the grid or something, but it seems hard.
        if self.illegal.is_some() {
            return Err(PARITY_SHADING_ILLEGAL_ACTION);
        }
        let rem = self.remaining.get_mut(&index);
        if let Some(r) = rem {
            if !r.contains(&value) {
                self.illegal = Some((index, value, self.shading_conflict_attr));
            } else {
                *r = VBitSet::singleton(&value);
            }
        }
        Ok(())
    }

    fn undo(&mut self, index: Index, value: StdVal<MIN, MAX>) -> Result<(), Error> {
        if let Some((i, v, _)) = self.illegal {
            if i != index || v != value {
                return Err(PARITY_SHADING_UNDO_MISMATCH);
            } else {
                self.illegal = None;
                return Ok(());
            }
        }
        let rem = self.remaining.get_mut(&index);
        if let Some(r) = rem {
            *r = self.remaining_init[&index].clone();
        }
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8, O: Overlay>
Constraint<StdVal<MIN, MAX>, O> for ParityShadingChecker<MIN, MAX> {
    fn name(&self) -> Option<String> { Some("ParityShading".to_string()) }

    fn check(&self, puzzle: &State<StdVal<MIN, MAX>, O>, ranking: &mut RankingInfo<StdVal<MIN, MAX>>) -> ConstraintResult<StdVal<MIN, MAX>> {
        if let Some((_, _, a)) = &self.illegal {
            return ConstraintResult::Contradiction(*a);
        }
        let grid = ranking.cells_mut();
        for (i, r) in &self.remaining {
            if let Some(_) = puzzle.get(*i) {
                continue;
            }
            let g = grid.get_mut(*i);
            g.0.intersect_with(r);
            g.1.add(&self.shading_feature, 1.0);
        }
        ConstraintResult::Ok
    }

    fn debug_at(&self, _: &State<StdVal<MIN, MAX>, O>, index: Index) -> Option<String> {
        let header = "ParityShadingChecker:\n";
        let mut lines = vec![];
        if let Some((i, v, a)) = &self.illegal {
            if *i == index {
                lines.push(format!("  Illegal move: {:?}={:?} ({})", i, v, a.name()));
            }
        }
        for (i, r) in &self.remaining {
            if *i != index {
                continue;
            }
            lines.push(format!("  - remaining vals: {}", r.to_string()));
        }
        if lines.is_empty() {
            None
        } else {
            Some(format!("{}{}", header, lines.join("\n")))
        }
    }

    fn debug_highlight(&self, _: &State<StdVal<MIN, MAX>, O>, index: Index) -> Option<(u8, u8, u8)> {
        self.colors.get(&index).map(|c| *c)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{constraint::{test_util::{assert_contradiction, assert_no_contradiction}, MultiConstraint}, ranker::StdRanker, solver::test_util::PuzzleReplay, sudoku::{four_standard_parse, StdChecker}};

    // This is a 4x4 puzzle with odd shadings on [0, 0] and [0, 2] and even
    // shadings on [0, 1] and [0, 3]. Call with different givens and an
    // expectation for it to return a contradiction (or not).
    fn assert_parity_shading_result(
        setup: &str, 
        expected: Option<&'static str>,
    ) {
        let pb = ParityShadingBuilder::new();
        let shadings = vec![
            pb.even([0, 0]),
            pb.odd([0, 1]),
            pb.even([0, 2]),
            pb.odd([0, 3]),
        ];
        let ranker = StdRanker::default();
        let mut puzzle = four_standard_parse(setup).unwrap();
        let mut constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(puzzle.overlay()),
            ParityShadingChecker::new(shadings),
        ]);
        let result = PuzzleReplay::new(&mut puzzle, &ranker, &mut constraint, None).replay().unwrap();
        if let Some(attr) = expected {
            assert_contradiction(result, attr);
        } else {
            assert_no_contradiction(result);
        }
    }

    #[test]
    fn test_parity_shading_conflict_even() {
        let input: &str = "1 .|. .\n\
                           . .|. .\n\
                           ---+---\n\
                           . .|. .\n\
                           . .|. .\n";
        assert_parity_shading_result(input, Some("PARITY_SHADING_CONFLICT"));
    }

    #[test]
    fn test_parity_shading_conflict_odd() {
        let input: &str = ". 2|. .\n\
                           . .|. .\n\
                           ---+---\n\
                           . .|. .\n\
                           . .|. .\n";
        assert_parity_shading_result(input, Some("PARITY_SHADING_CONFLICT"));
    }

    #[test]
    fn test_parity_shading_infeasible_interaction() {
        // [0, 0] needs to be even, but no even values are left
        let input: &str = ". .|. .\n\
                           . .|. .\n\
                           ---+---\n\
                           2 .|. .\n\
                           4 .|. .\n";
        assert_parity_shading_result(input, Some("DG_CELL_NO_VALS"));
    }

    #[test]
    fn test_parity_shading_valid_fill() {
        let input: &str = "2 1|4 3\n\
                           . .|. .\n\
                           ---+---\n\
                           . .|. .\n\
                           . .|. .\n";
        assert_parity_shading_result(input, None);
    }
}