use std::{collections::HashMap, fmt::Debug};
use crate::{constraint::Constraint, core::{Attribution, ConstraintResult, Error, Feature, Index, Key, Overlay, State, Stateful, VSetMut}, index_util::{check_adjacent, expand_polyline}, range_util::Range, ranker::RankingInfo, sudoku::StdVal};

// TODO: Add support for slow thermos

#[derive(Debug, Clone)]
pub struct Thermo {
    pub cells: Vec<Index>,
}

impl Thermo {
    pub fn contains(&self, index: Index) -> bool {
        self.cells.contains(&index)
    }
}

pub struct ThermoBuilder<const MIN: u8, const MAX: u8>;
impl <const MIN: u8, const MAX: u8> ThermoBuilder<MIN, MAX> {
    pub fn new() -> Self { Self }

    pub fn thermo(&self, cells: Vec<Index>) -> Thermo {
        let max_len: usize = (MAX as usize) + 1 - (MIN as usize);
        if cells.len() > max_len {
            panic!("Thermo too long: {} cells, max is {}", cells.len(), max_len);
        }
        for (i, &cell) in cells.iter().enumerate() {
            if i > 0 {
                check_adjacent(cells[i - 1], cell).unwrap();
            }
        }
        Thermo { cells }
    }

    pub fn right(&self, left: Index, length: usize) -> Thermo {
        let cells = (0..length)
            .map(|i| [left[0], left[1]+i])
            .collect();
        self.thermo(cells)
    }

    pub fn left(&self, right: Index, length: usize) -> Thermo {
        let cells = (0..length)
            .map(|i| [right[0], right[1]-i])
            .collect();
        self.thermo(cells)
    }

    pub fn down(&self, top: Index, length: usize) -> Thermo {
        let cells = (0..length)
            .map(|i| [top[0]+i, top[1]])
            .collect();
        self.thermo(cells)
    }

    pub fn up(&self, bottom: Index, length: usize) -> Thermo {
        let cells = (0..length)
            .map(|i| [bottom[0]-i, bottom[1]])
            .collect();
        self.thermo(cells)
    }

    pub fn polyline(&self, vertices: Vec<Index>) -> Thermo {
        self.thermo(expand_polyline(vertices).unwrap())
    }
}

pub const THERMO_ILLEGAL_ACTION: Error = Error::new_const("A thermo violation already exists; can't apply further actions.");
pub const THERMO_UNDO_MISMATCH: Error = Error::new_const("Undo value mismatch");
pub const THERMO_FEATURE: &str = "THERMO";
pub const THERMO_BULB_FEATURE: &str = "THERMO_BULB";
//pub const THERMO_CONFLICT_ATTRIBUTION: &str = "THERMO_CONFLICT";

pub struct ThermoChecker<const MIN: u8, const MAX: u8> {
    thermos: Vec<Thermo>,
    init_ranges: HashMap<Index, Range<MIN, MAX>>,
    thermo_feature: Key<Feature>,
    thermo_bulb_feature: Key<Feature>,
    //thermo_conflict_attr: Key<Attribution>,
    illegal: Option<(Index, StdVal<MIN, MAX>, Key<Attribution>)>,
}

fn thermo_init<const MIN: u8, const MAX: u8>(thermo: &Thermo) -> Vec<(Index, Range<MIN, MAX>)> {
    let len = thermo.cells.len() as u8;
    thermo.cells.iter().enumerate().map(|(i, c)| {
        (*c, Range::HalfOpen(MIN+i as u8, MAX+2-len+i as u8))
    }).collect()
}

fn thermo_calc<const MIN: u8, const MAX: u8>(
    thermo: &Thermo,
    init_ranges: &HashMap<Index, Range<MIN, MAX>>,
    ranges_from_grid: Vec<Range<MIN, MAX>>,
) -> Vec<Range<MIN, MAX>> {
    let mut res: Vec<Range<MIN, MAX>> = thermo.cells
        .iter().enumerate().map(|(i, c)| {
        init_ranges.get(c).unwrap().intersection(&ranges_from_grid[i])
    }).collect();
    let len = res.len();
    for i in 0..len {
        if let Range::HalfOpen(min, max) = res[i] {
            // Restrict upwards 
            for j in (i+1)..len {
                res[j].clip_min(if min + (j as u8 - i as u8) <= MAX {
                    min + (j as u8 - i as u8)
                } else {
                    MAX
                })
            }
            // Restrict downwards
            for j in 0..i {
                res[j].clip_max(if MIN + (i as u8 - j as u8) >= max {
                    MIN
                } else {
                    max - (i as u8 - j as u8)
                })
            }
        }
    }
    res
}

impl <const MIN: u8, const MAX: u8> ThermoChecker<MIN, MAX> {
    pub fn new(thermos: Vec<Thermo>) -> Self {
        let mut init_ranges: HashMap<Index, Range<MIN, MAX>> = HashMap::new();
        for t in &thermos {
            for (i, r1) in thermo_init::<MIN, MAX>(t) {
                init_ranges.entry(i)
                    .and_modify(|r2| {
                        *r2 = r1.intersection(r2)
                    })
                    .or_insert(r1);
            }
        }
        Self {
            thermos,
            init_ranges,
            thermo_feature: Key::register(THERMO_FEATURE),
            thermo_bulb_feature: Key::register(THERMO_BULB_FEATURE),
            //thermo_conflict_attr: Key::register(THERMO_CONFLICT_ATTRIBUTION),
            illegal: None,
        }
    }
}

impl <const MIN: u8, const MAX: u8> Debug for ThermoChecker<MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((i, v, a)) = &self.illegal {
            write!(f, "Illegal move: {:?}={:?} ({})\n", i, v, a.name())?;
        }
        for t in &self.thermos {
            write!(f, " Thermo{:?}\n", t.cells)?;
            // TODO: Show something more useful?
        }
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8> Stateful<StdVal<MIN, MAX>> for ThermoChecker<MIN, MAX> {
    fn reset(&mut self) {
        self.illegal = None;
    }

    fn apply(&mut self, index: Index, _: StdVal<MIN, MAX>) -> Result<(), Error> {
        // In theory we could be allow multiple illegal moves and just
        // invalidate and recalculate the grid or something, but it seems hard.
        if self.illegal.is_some() {
            return Err(THERMO_ILLEGAL_ACTION);
        }
        for (_, t) in self.thermos.iter().enumerate() {
            if !t.contains(index) {
                continue;
            }
            // TODO: Keep track of values and check for an immediate violation.
        }
        Ok(())
    }

    fn undo(&mut self, index: Index, value: StdVal<MIN, MAX>) -> Result<(), Error> {
        if let Some((i, v, _)) = self.illegal {
            if i != index || v != value {
                return Err(THERMO_UNDO_MISMATCH);
            } else {
                self.illegal = None;
                return Ok(());
            }
        }
        // TODO: Keep track of values.
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8, O: Overlay>
Constraint<StdVal<MIN, MAX>, O> for ThermoChecker<MIN, MAX> {
    fn name(&self) -> Option<String> { Some("ThermoChecker".to_string()) }
    fn check(&self, _: &State<StdVal<MIN, MAX>, O>, ranking: &mut RankingInfo<StdVal<MIN, MAX>>) -> ConstraintResult<StdVal<MIN, MAX>> {
        if let Some((_, _, a)) = &self.illegal {
            return ConstraintResult::Contradiction(*a);
        }
        let grid = ranking.cells_mut();
        for t in &self.thermos {
            grid.get_mut(t.cells[0]).1.add(&self.thermo_bulb_feature, 1.0);
            let ranges_from_grid = t.cells.iter().map(|c| {
                Range::from_set(grid.get(*c).0)
            }).collect();
            for (i, r) in thermo_calc(t, &self.init_ranges, ranges_from_grid).into_iter().enumerate() {
                let g = grid.get_mut(t.cells[i]);
                g.0.intersect_with(&r.to_set());
                g.1.add(&self.thermo_feature, 1.0);
            }
        }
        ConstraintResult::Ok
    }

    fn debug_at(&self, _: &State<StdVal<MIN, MAX>, O>, index: Index) -> Option<String> {
        let header = "ThermoChecker:\n";
        let mut lines = vec![];
        if let Some((i, v, a)) = &self.illegal {
            if *i == index {
                lines.push(format!("  Illegal move: {:?}={:?} ({})", i, v, a.name()));
            }
            // TODO: Provide useful info for the relevant thermos
        }
        if lines.is_empty() {
            None
        } else {
            Some(format!("{}{}", header, lines.join("\n")))
        }
    }

    fn debug_highlight(&self, _: &State<StdVal<MIN, MAX>, O>, index: Index) -> Option<(u8, u8, u8)> {
        if let Some((i, _, _)) = &self.illegal {
            if *i == index {
                return Some((200, 0, 0));
            }
        }
        for t in &self.thermos {
            if t.cells[0] == index {
                return Some((200, 200, 0))
            } else if t.contains(index) {
                return Some((150, 150, 0))
            }
        }
        None
    }
}

#[cfg(test)]
mod test {
    use crate::{constraint::MultiConstraint, ranker::StdRanker, solver::{FindFirstSolution, PuzzleSetter}, sudoku::{four_standard_parse, FourStd, FourStdOverlay, FourStdVal, StdChecker}};

    use super::*;

    #[test]
    fn test_thermo_init() {
        let tb = ThermoBuilder::<1, 9>::new();
        assert_eq!(
            thermo_init::<1, 9>(&tb.right([0, 0], 8)),
            vec![
                ([0, 0], Range::HalfOpen(1, 3)),
                ([0, 1], Range::HalfOpen(2, 4)),
                ([0, 2], Range::HalfOpen(3, 5)),
                ([0, 3], Range::HalfOpen(4, 6)),
                ([0, 4], Range::HalfOpen(5, 7)),
                ([0, 5], Range::HalfOpen(6, 8)),
                ([0, 6], Range::HalfOpen(7, 9)),
                ([0, 7], Range::HalfOpen(8, 10)),
            ],
        );
        assert_eq!(
            thermo_init::<1, 9>(&tb.right([0, 0], 3)),
            vec![
                ([0, 0], Range::HalfOpen(1, 8)),
                ([0, 1], Range::HalfOpen(2, 9)),
                ([0, 2], Range::HalfOpen(3, 10)),
            ],
        );
    }

    #[test]
    fn test_thermo_calc() {
        let t = ThermoBuilder::<1, 9>::new().right([0, 0], 6);
        // A thermo of length six initially has ranges like this:
        // [1, 5), [2, 6), [3, 7), [4, 8), [5, 9), [6, 10)
        //          ^                  ^
        //          4                  7
        // We'll make the above updates to those ranges and it should
        // propagate outwards to this result:
        // [1, 4), [4, 5), [5, 6), [6, 7), [7, 9), [8, 10)
        let mut init_ranges = HashMap::new();
        for (i, r) in thermo_init::<1, 9>(&t) {
            init_ranges.insert(i, r);
        }
        let trivial = Range::<1, 9>::HalfOpen(1, 10);
        let ranges_from_grid = vec![
            trivial,
            // Bring the min for this up by 2
            Range::HalfOpen(4, 10),
            trivial,
            // Bring the max for this down by 1
            Range::HalfOpen(1, 7),
            trivial,
            trivial,
        ];
        assert_eq!(
            thermo_calc(&t, &init_ranges, ranges_from_grid),
            vec![
                Range::HalfOpen(1, 4),
                Range::HalfOpen(4, 5),
                Range::HalfOpen(5, 6),
                Range::HalfOpen(6, 7),
                Range::HalfOpen(7, 9),
                Range::HalfOpen(8, 10),
            ],
        )
    }

    // https://www.sporcle.com/games/HerrieM/thermo-sudoku-1
    struct E2EThermo;
    impl PuzzleSetter for E2EThermo {
        type Value = FourStdVal;
        type Overlay = FourStdOverlay;
        type Ranker = StdRanker;
        type Constraint = MultiConstraint<Self::Value, Self::Overlay>;
        fn setup() -> (FourStd, Self::Ranker, Self::Constraint) {
            Self::setup_with_givens(four_standard_parse(
                "2 .|. .\n\
                 . .|. .\n\
                 ---+---\n\
                 . .|. 2\n\
                 . .|. .\n"
            ).unwrap())
        }
        fn setup_with_givens(given: FourStd) -> (FourStd, Self::Ranker, Self::Constraint) {
            let tb = ThermoBuilder::<1, 4>::new();
            let thermos = vec![
                tb.right([1, 1], 3),
                tb.right([2, 0], 3),
                tb.left([3, 2], 3),
            ];
            let constraint = MultiConstraint::new(vec_box::vec_box![
                StdChecker::new(&given),
                ThermoChecker::new(thermos),
            ]);
            (given, StdRanker::default(), constraint)
        }
    }

    #[test]
    fn test_e2e_thermo_example() -> Result<(), Error> {
        let (mut puzzle, ranker, mut constraint) = E2EThermo::setup();
        let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, None);
        let maybe_solution = finder.solve()?;
        assert!(maybe_solution.is_some());
        let expected: &str = "2 4|3 1\n\
                              3 1|2 4\n\
                              ---+---\n\
                              1 3|4 2\n\
                              4 2|1 3\n";
        let solution = maybe_solution.unwrap().state();
        assert_eq!(solution.overlay().serialize_pretty(&solution), expected);
        Ok(())
    }
}