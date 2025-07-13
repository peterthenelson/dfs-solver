use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::{LazyLock, Mutex};
use crate::color_util::{color_fib_palette, color_opt_ave2, color_planar_graph, find_undirected_edges};
use crate::constraint::Constraint;
use crate::core::{Attribution, ConstraintResult, Feature, Index, Key, Overlay, State, Stateful, VBitSet, VBitSetRef, VSet, VSetMut, Value};
use crate::illegal_move::IllegalMove;
use crate::index_util::{check_orthogonally_adjacent, collections_orthogonally_neighboring, expand_orthogonal_polyline};
use crate::memo::{FnToCalc, MemoLock};
use crate::ranker::RankingInfo;
use crate::sudoku::{unpack_stdval_vals, StdOverlay, StdVal};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum KropkiColor { Black, White }

#[derive(Debug, Clone)]
pub struct KropkiDotChain {
    color: KropkiColor,
    cells: Vec<Index>,
    mutually_visible: bool,
}

impl KropkiDotChain {
    pub fn contains(&self, index: Index) -> bool {
        self.cells.contains(&index)
    }
}

pub struct KropkiBuilder<'a, O: Overlay>(&'a O);

impl <'a, O: Overlay> KropkiBuilder<'a, O> {
    pub fn new(overlay: &'a O) -> Self { Self(overlay) }

    fn create(&self, color: KropkiColor, cells: Vec<Index>) -> KropkiDotChain {
        for (i, &cell) in cells.iter().enumerate() {
            if i > 0 {
                check_orthogonally_adjacent(cells[i - 1], cell).unwrap();
            }
            if i < cells.len() - 1{
                check_orthogonally_adjacent(cell, cells[i + 1]).unwrap();
            }
        }
        let mutually_visible = self.0.all_mutually_visible(&cells);
        KropkiDotChain {
            color,
            cells,
            mutually_visible,
        }
    }

    pub fn b_across(&self, left: Index) -> KropkiDotChain {
        self.create(KropkiColor::Black, vec![left, [left[0], left[1]+1]])
    }

    pub fn w_across(&self, left: Index) -> KropkiDotChain {
        self.create(KropkiColor::White, vec![left, [left[0], left[1]+1]])
    }

    pub fn b_down(&self, top: Index) -> KropkiDotChain {
        self.create(KropkiColor::Black, vec![top, [top[0]+1, top[1]]])
    }

    pub fn w_down(&self, top: Index) -> KropkiDotChain {
        self.create(KropkiColor::White, vec![top, [top[0]+1, top[1]]])
    }

    pub fn b_chain(&self, cells: Vec<Index>) -> KropkiDotChain {
        self.create(KropkiColor::Black, cells)
    }

    pub fn w_chain(&self, cells: Vec<Index>) -> KropkiDotChain {
        self.create(KropkiColor::White, cells)
    }

    pub fn b_polyline(&self, vertices: Vec<Index>) -> KropkiDotChain {
        self.create(KropkiColor::Black, expand_orthogonal_polyline(vertices).unwrap())
    }

    pub fn w_polyline(&self, vertices: Vec<Index>) -> KropkiDotChain {
        self.create(KropkiColor::White, expand_orthogonal_polyline(vertices).unwrap())
    }
}

/// Tables of useful sets for kropki black dot constraints.
static KB_POSSIBLE: LazyLock<Mutex<HashMap<(u8, u8), bit_set::BitSet>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});
static KB_POSSIBLE_MV: LazyLock<Mutex<HashMap<(u8, u8, usize), Vec<bit_set::BitSet>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

fn kb_possible_raw<const MIN: u8, const MAX: u8>(_: &()) -> bit_set::BitSet {
    let mut set = VBitSet::<StdVal<MIN, MAX>>::empty();
    for v in StdVal::<MIN, MAX>::possibilities() {
        if v.val() % 2 == 0 {
            let half = v.val() / 2;
            if MIN <= half {
                set.insert(&v);
                continue;
            }
        }
        if v.val() < 128 {
            let double = v.val() * 2;
            if double <= MAX {
                set.insert(&v);
                continue;
            }
        }
    }
    set.into_erased()
}

pub struct KBPossible<const MIN: u8, const MAX: u8>(MemoLock<(), (u8, u8), bit_set::BitSet, FnToCalc<(), (u8, u8), bit_set::BitSet>>);
impl <const MIN: u8, const MAX: u8> KBPossible<MIN, MAX> {
    pub fn get(&mut self) -> VBitSetRef<StdVal<MIN, MAX>> {
        VBitSetRef::assume_typed(self.0.get(&()))
    }
}

pub fn kropki_black_possible<const MIN: u8, const MAX: u8>() -> KBPossible<MIN, MAX> {
    let guard = KB_POSSIBLE.lock().unwrap();
    let calc = FnToCalc::<_, _, _>::new(
        |&()| (MIN, MAX),
        kb_possible_raw::<MIN, MAX>,
    );
    KBPossible(MemoLock::new(guard, calc))
}

fn kb_possible_mv_raw<const MIN: u8, const MAX: u8>(args: &(usize,)) -> Vec<bit_set::BitSet> {
    let n_mutually_visible = args.0;
    let mut possible = vec![VBitSet::<StdVal<MIN, MAX>>::empty(); n_mutually_visible/2 + n_mutually_visible % 2];
    for v in StdVal::<MIN, MAX>::possibilities() {
        let mut chain: Vec<StdVal<MIN, MAX>> = vec![];
        let mut cur = v.val() as u16;
        while cur <= (MAX as u16) && chain.len() < n_mutually_visible {
            chain.push(StdVal::new(cur as u8));
            cur *= 2;
        }
        if chain.len() != n_mutually_visible {
            continue;
        }
        for i in 0..possible.len() {
            possible[i].insert(&chain[i]);
            possible[i].insert(&chain[chain.len()-1-i]);
        }
    }
    possible.into_iter().map(|v| v.into_erased()).collect()
}

pub struct KBPossibleChain<const MIN: u8, const MAX: u8>(MemoLock<(usize,), (u8, u8, usize), Vec<bit_set::BitSet>, FnToCalc<(usize,), (u8, u8, usize), Vec<bit_set::BitSet>>>);
impl <const MIN: u8, const MAX: u8> KBPossibleChain<MIN, MAX> {
    pub fn get(&mut self, n_mutually_visible: usize, mut len_from_end: usize) -> VBitSetRef<StdVal<MIN, MAX>> {
        if n_mutually_visible < 2 {
            panic!("kropki_black_possible_chain only makes sense when chain length \
                    is at least 2; got {}", n_mutually_visible);
        }
        if len_from_end >= n_mutually_visible {
            panic!("kropki_black_possible_chain length argument can't be as high \
                    or higher than the length of the chain itself; got {}", len_from_end)
        } else if len_from_end >= (n_mutually_visible / 2) {
            // Standardize to the smaller of two equiv values
            len_from_end = n_mutually_visible - 1 - len_from_end;
        }
        VBitSetRef::<StdVal<MIN, MAX>>::assume_typed(
            &self.0.get(&(n_mutually_visible,))[len_from_end],
        )
    }
}

pub fn kropki_black_possible_chain<const MIN: u8, const MAX: u8>() -> KBPossibleChain<MIN, MAX> {
    let guard = KB_POSSIBLE_MV.lock().unwrap();
    let calc = FnToCalc::<_, _, _>::new(
        |&(n_mutually_visible,)| (MIN, MAX, n_mutually_visible),
        kb_possible_mv_raw::<MIN, MAX>,
    );
    KBPossibleChain(MemoLock::new(guard, calc))
}

fn kropki_black_adj_ok<const MIN: u8, const MAX: u8, VS: VSet<StdVal<MIN, MAX>>>(a: &VS, b: &VS) -> bool {
    for v1 in unpack_stdval_vals::<MIN, MAX, _>(a) {
        for v2 in unpack_stdval_vals::<MIN, MAX, _>(b) {
            if (v1 < v2 && v1 < 128 && v1*2 == v2) ||
               (v2 < v1 && v2 < 128 && v2*2 == v1) {
                return true;
            }
        }
    }
    false
}

fn get_black_lower<const MIN: u8, const MAX: u8>(v: StdVal<MIN, MAX>) -> Option<StdVal<MIN, MAX>> {
    if v.val() % 2 == 0 && v.val()/2 >= MIN {
        Some(StdVal::new(v.val()/2))
    } else {
        None
    }
}

fn get_black_upper<const MIN: u8, const MAX: u8>(v: StdVal<MIN, MAX>) -> Option<StdVal<MIN, MAX>> {
    if v.val() < 128 && v.val()*2 <= MAX {
        Some(StdVal::new(v.val()*2))
    } else {
        None
    }
}

fn kropki_black_between<const MIN: u8, const MAX: u8, VS: VSet<StdVal<MIN, MAX>>>(
    left: &VS, right: &Option<VS>, mutually_visible: bool,
) -> VBitSet<StdVal<MIN, MAX>> {
    let mut possible = VBitSet::<StdVal<MIN, MAX>>::empty();
    if let Some(right) = right {
        for v in StdVal::<MIN, MAX>::possibilities() {
            let (lower, upper) = (get_black_lower::<MIN, MAX>(v), get_black_upper::<MIN, MAX>(v));
            if mutually_visible {
                if lower.is_none() || upper.is_none() {
                    continue;
                }
                let (l, u) = (lower.unwrap(), upper.unwrap());
                if (left.contains(&l) && right.contains(&u)) ||
                (left.contains(&u) && right.contains(&l)) {
                    possible.insert(&v);
                }
            } else if lower.is_some() && upper.is_some() {
                let (l, u) = (lower.unwrap(), upper.unwrap());
                if (left.contains(&l) || left.contains(&u)) &&
                (right.contains(&l) || right.contains(&u)) {
                    possible.insert(&v);
                }
            } else if let Some(l) = lower {
                if left.contains(&l) && right.contains(&l) {
                    possible.insert(&v);
                }
            } else if let Some(u) = upper {
                if left.contains(&u) && right.contains(&u) {
                    possible.insert(&v);
                }
            }
        }
    } else {
        for v in StdVal::<MIN, MAX>::possibilities() {
            let (lower, upper) = (get_black_lower::<MIN, MAX>(v), get_black_upper::<MIN, MAX>(v));
            if let Some(l) = lower {
                if left.contains(&l) {
                    possible.insert(&v);
                }
            }
            if let Some(u) = upper {
                if left.contains(&u) {
                    possible.insert(&v);
                }
            }
        }
    }
    possible
}

/// Table of useful sets for kropki white dot constraints.
static KW_POSSIBLE_MV: LazyLock<Mutex<HashMap<(u8, u8, usize), Vec<bit_set::BitSet>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

fn kw_possible_mv_raw<const MIN: u8, const MAX: u8>(args: &(usize,)) -> Vec<bit_set::BitSet> {
    let n_mutually_visible = args.0;
    let mut possible = vec![VBitSet::<StdVal<MIN, MAX>>::empty(); n_mutually_visible/2 + n_mutually_visible % 2];
    for v in StdVal::<MIN, MAX>::possibilities() {
        let mut chain: Vec<StdVal<MIN, MAX>> = vec![];
        let mut cur = v.val() as u16;
        while cur <= (MAX as u16) && chain.len() < n_mutually_visible {
            chain.push(StdVal::new(cur as u8));
            cur += 1;
        }
        if chain.len() != n_mutually_visible {
            continue;
        }
        for i in 0..possible.len() {
            possible[i].insert(&chain[i]);
            possible[i].insert(&chain[chain.len()-1-i]);
        }
    }
    possible.into_iter().map(|v| v.into_erased()).collect()
}

pub struct KWPossibleChain<const MIN: u8, const MAX: u8>(MemoLock<(usize,), (u8, u8, usize), Vec<bit_set::BitSet>, FnToCalc<(usize,), (u8, u8, usize), Vec<bit_set::BitSet>>>);
impl <const MIN: u8, const MAX: u8> KWPossibleChain<MIN, MAX> {
    pub fn get(&mut self, n_mutually_visible: usize, mut len_from_end: usize) -> VBitSetRef<StdVal<MIN, MAX>> {
        if n_mutually_visible < 2 {
            panic!("kropki_white_possible_chain only makes sense when chain length \
                    is at least 2; got {}", n_mutually_visible);
        }
        if len_from_end >= n_mutually_visible {
            panic!("kropki_white_possible_chain length argument can't be as high \
                    or higher than the length of the chain itself; got {}", len_from_end)
        } else if len_from_end >= (n_mutually_visible / 2) {
            // Standardize to the smaller of two equiv values
            len_from_end = n_mutually_visible - 1 - len_from_end;
        }
        VBitSetRef::<StdVal<MIN, MAX>>::assume_typed(
            &self.0.get(&(n_mutually_visible,))[len_from_end],
        )
    }
}

pub fn kropki_white_possible_chain<const MIN: u8, const MAX: u8>() -> KWPossibleChain<MIN, MAX> {
    let guard = KW_POSSIBLE_MV.lock().unwrap();
    let calc = FnToCalc::<_, _, _>::new(
        |&(n_mutually_visible,)| (MIN, MAX, n_mutually_visible),
        kw_possible_mv_raw::<MIN, MAX>,
    );
    KWPossibleChain(MemoLock::new(guard, calc))
}

fn kropki_white_adj_ok<const MIN: u8, const MAX: u8, VS: VSet<StdVal<MIN, MAX>>>(a: &VS, b: &VS) -> bool {
    for v1 in unpack_stdval_vals::<MIN, MAX, _>(a) {
        for v2 in unpack_stdval_vals::<MIN, MAX, _>(b) {
            if v1.abs_diff(v2) == 1 {
                return true;
            }
        }
    }
    false
}

fn get_white_lower<const MIN: u8, const MAX: u8>(v: StdVal<MIN, MAX>) -> Option<StdVal<MIN, MAX>> {
    if v.val() > MIN {
        Some(StdVal::<MIN, MAX>::new(v.val() - 1))
    } else {
        None
    }
}

fn get_white_upper<const MIN: u8, const MAX: u8>(v: StdVal<MIN, MAX>) -> Option<StdVal<MIN, MAX>> {
    if v.val() < MAX {
        Some(StdVal::<MIN, MAX>::new(v.val() + 1))
    } else {
        None
    }
}

fn kropki_white_between<const MIN: u8, const MAX: u8, VS: VSet<StdVal<MIN, MAX>>>(
    left: &VS, right: &Option<VS>, mutually_visible: bool,
) -> VBitSet<StdVal<MIN, MAX>> {
    let mut possible = VBitSet::<StdVal<MIN, MAX>>::empty();
    if let Some(right) = right {
        for v in StdVal::<MIN, MAX>::possibilities() {
            let (lower, upper) = (get_white_lower::<MIN, MAX>(v), get_white_upper::<MIN, MAX>(v));
            if mutually_visible {
                if lower.is_none() || upper.is_none() {
                    continue;
                }
                let (l, u) = (lower.unwrap(), upper.unwrap());
                if (left.contains(&l) && right.contains(&u)) ||
                (left.contains(&u) && right.contains(&l)) {
                    possible.insert(&v);
                }
            } else if lower.is_some() && upper.is_some() {
                let (l, u) = (lower.unwrap(), upper.unwrap());
                if (left.contains(&l) || left.contains(&u)) &&
                (right.contains(&l) || right.contains(&u)) {
                    possible.insert(&v);
                }
            } else if let Some(l) = lower {
                if left.contains(&l) && right.contains(&l) {
                    possible.insert(&v);
                }
            } else if let Some(u) = upper {
                if left.contains(&u) && right.contains(&u) {
                    possible.insert(&v);
                }
            }
        }
    } else {
        for v in StdVal::<MIN, MAX>::possibilities() {
            let (lower, upper) = (get_white_lower::<MIN, MAX>(v), get_white_upper::<MIN, MAX>(v));
            if let Some(l) = lower {
                if left.contains(&l) {
                    possible.insert(&v);
                }
            }
            if let Some(u) = upper {
                if left.contains(&u) {
                    possible.insert(&v);
                }
            }
        }
    }
    possible
}

pub const KROPKI_BLACK_FEATURE: &str = "KROPKI_BLACK";
pub const KROPKI_BLACK_CONFLICT_ATTRIBUTION: &str = "KROPKI_BLACK_CONFLICT";
pub const KROPKI_BLACK_INFEASIBLE_ATTRIBUTION: &str = "KROPKI_BLACK_INFEASIBLE";
pub const KROPKI_WHITE_FEATURE: &str = "KROPKI_WHITE";
pub const KROPKI_WHITE_CONFLICT_ATTRIBUTION: &str = "KROPKI_WHITE_CONFLICT";
pub const KROPKI_WHITE_INFEASIBLE_ATTRIBUTION: &str = "KROPKI_WHITE_INFEASIBLE";

fn check_coverage_and_add_color(chains: &Vec<KropkiDotChain>, color: KropkiColor) -> Vec<(u8, u8, u8)> {
    let mut covered = HashSet::new();
    for c in chains {
        for cell in c.cells.iter() {
            if covered.contains(&cell) {
                let color_s = match color {
                    KropkiColor::Black => "black",
                    KropkiColor::White => "white",
                };
                panic!("Multiple {} kropki chains contain cell: {:?}\n", color_s, cell);
            }
            covered.insert(cell);
        }
    }
    let edges = find_undirected_edges(&chains, |c1, c2| {
        collections_orthogonally_neighboring(&c1.cells, &c2.cells)
    });
    let palette = color_fib_palette((200, 200, 0), 5, 50.0);
    color_planar_graph(edges, &palette)
}

pub struct KropkiChecker<const MIN: u8, const MAX: u8> {
    blacks: Vec<KropkiDotChain>,
    black_remaining: HashMap<Index, VBitSet<StdVal<MIN, MAX>>>,
    black_colors: Vec<(u8, u8, u8)>,
    kb_feature: Key<Feature>,
    kb_conflict_attr: Key<Attribution>,
    kb_if_attr: Key<Attribution>,
    whites: Vec<KropkiDotChain>,
    white_remaining: HashMap<Index, VBitSet<StdVal<MIN, MAX>>>,
    white_colors: Vec<(u8, u8, u8)>,
    kw_feature: Key<Feature>,
    kw_conflict_attr: Key<Attribution>,
    kw_if_attr: Key<Attribution>,
    illegal: IllegalMove<StdVal<MIN, MAX>>,
}

impl <const MIN: u8, const MAX: u8> KropkiChecker<MIN, MAX> {
    pub fn new(chains: Vec<KropkiDotChain>) -> Self {
        let (blacks, whites): (Vec<KropkiDotChain>, Vec<_>) = chains
            .into_iter()
            .partition(|c| c.color == KropkiColor::Black);
        let black_colors = check_coverage_and_add_color(&blacks, KropkiColor::Black);
        let white_colors = check_coverage_and_add_color(&whites, KropkiColor::White);
        let mut kc = Self {
            blacks,
            black_remaining: HashMap::new(),
            black_colors,
            kb_feature: Key::register(KROPKI_BLACK_FEATURE),
            kb_conflict_attr: Key::register(KROPKI_BLACK_CONFLICT_ATTRIBUTION),
            kb_if_attr: Key::register(KROPKI_BLACK_INFEASIBLE_ATTRIBUTION),
            whites,
            white_remaining: HashMap::new(),
            white_colors,
            kw_feature: Key::register(KROPKI_WHITE_FEATURE),
            kw_conflict_attr: Key::register(KROPKI_WHITE_CONFLICT_ATTRIBUTION),
            kw_if_attr: Key::register(KROPKI_WHITE_INFEASIBLE_ATTRIBUTION),
            illegal: IllegalMove::new(),
        };
        kc.reset();
        kc
    }
}

impl <const MIN: u8, const MAX: u8> Debug for KropkiChecker<MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.illegal.write_dbg(f)?;
        for (i, b) in self.blacks.iter().enumerate() {
            write!(f, " Black[{}]: ", i)?;
            for cell in &b.cells {
                let rem = self.black_remaining.get(cell)
                    .expect(format!("remaining[{:?}] not found!", cell).as_str());
                write!(f, "{:?}=>{:?} ", cell, unpack_stdval_vals::<MIN, MAX, _>(rem))?;
            }
            write!(f, "\n")?;
        }
        for (i, w) in self.whites.iter().enumerate() {
            write!(f, " White[{}]: ", i)?;
            for cell in &w.cells {
                let rem = self.white_remaining.get(cell)
                    .expect(format!("remaining[{:?}] not found!", cell).as_str());
                write!(f, "{:?}=>{:?} ", cell, unpack_stdval_vals::<MIN, MAX, _>(rem))?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8> Stateful<StdVal<MIN, MAX>> for KropkiChecker<MIN, MAX> {
    fn reset(&mut self) {
        for b in self.blacks.iter() {
            if b.mutually_visible {
                for (i, cell) in b.cells.iter().enumerate() {
                    self.black_remaining.insert(
                        *cell,
                        kropki_black_possible_chain::<MIN, MAX>().get(b.cells.len(), i).to_vbitset(),
                    );
                }
            } else {
                for cell in &b.cells {
                    self.black_remaining.insert(
                        *cell, 
                        kropki_black_possible::<MIN, MAX>().get().to_vbitset()
                    );
                }
            }
        }
        for w in self.whites.iter() {
            if w.mutually_visible {
                for (i, cell) in w.cells.iter().enumerate() {
                    self.white_remaining.insert(
                        *cell,
                        kropki_white_possible_chain::<MIN, MAX>().get(w.cells.len(), i).to_vbitset(),
                    );
                }
            } else {
                for cell in &w.cells {
                    self.white_remaining.insert(
                        *cell, 
                        VBitSet::<StdVal<MIN, MAX>>::full(),
                    );
                }
            }
        }
        self.illegal.reset();
    }

    // TODO: We could be smarter about white chains (and share some logic with
    // the thermos).
    fn apply(&mut self, index: Index, value: StdVal<MIN, MAX>) -> Result<(), crate::core::Error> {
        self.illegal.check_unset()?;
        let maybe_b_rem = if let Some(r) = self.black_remaining.get_mut(&index) {
            if !r.contains(&value) {
                self.illegal.set(index, value, self.kb_conflict_attr);
                return Ok(());
            }
            Some(r)
        } else {
            None
        };
        let maybe_w_rem = if let Some(r) = self.white_remaining.get_mut(&index) {
            if !r.contains(&value) {
                self.illegal.set(index, value, self.kw_conflict_attr);
                return Ok(());
            }
            Some(r)
        } else {
            None
        };
        if let Some(r) = maybe_b_rem {
            *r = VBitSet::<StdVal<MIN, MAX>>::singleton(&value);
        }
        if let Some(r) = maybe_w_rem {
            *r = VBitSet::<StdVal<MIN, MAX>>::singleton(&value);
        }
        Ok(())
    }

    fn undo(&mut self, index: Index, value: StdVal<MIN, MAX>) -> Result<(), crate::core::Error> {
        if self.illegal.undo(index, value)? {
            return Ok(());
        }
        if self.black_remaining.contains_key(&index) {
            for b in &self.blacks {
                for (i, cell) in b.cells.iter().enumerate() {
                    if *cell != index {
                        continue;
                    }
                    *self.black_remaining.get_mut(cell).unwrap() = if b.mutually_visible {
                        kropki_black_possible_chain::<MIN, MAX>().get(b.cells.len(), i).to_vbitset()
                    } else {
                        kropki_black_possible::<MIN, MAX>().get().to_vbitset()
                    };
                    break;
                }
            }
        }
        if self.white_remaining.contains_key(&index) {
            for w in &self.whites {
                for (i, cell) in w.cells.iter().enumerate() {
                    if *cell != index {
                        continue;
                    }
                    *self.white_remaining.get_mut(cell).unwrap() = if w.mutually_visible {
                        kropki_white_possible_chain::<MIN, MAX>().get(w.cells.len(), i).to_vbitset()
                    } else {
                        VBitSet::<StdVal<MIN, MAX>>::full()
                    };
                    break;
                }
            }
        }
        Ok(())
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8>
Constraint<StdVal<MIN, MAX>, StdOverlay<N, M>> for KropkiChecker<MIN, MAX> {
    fn name(&self) -> Option<String> { Some("KropkiChecker".to_string()) }
    fn check(&self, _: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, ranking: &mut RankingInfo<StdVal<MIN, MAX>>) -> ConstraintResult<StdVal<MIN, MAX>> {
        if let Some(c) = self.illegal.to_contradiction() {
            return c;
        }
        let grid = ranking.cells_mut();
        for b in &self.blacks {
            for cell in &b.cells {
                let g = grid.get_mut(*cell);
                g.1.add(&self.kb_feature, 1.0);
                g.0.intersect_with(self.black_remaining.get(cell).unwrap());
            }
        }
        for w in &self.whites {
            for cell in &w.cells {
                let g = grid.get_mut(*cell);
                g.1.add(&self.kw_feature, 1.0);
                g.0.intersect_with(self.white_remaining.get(cell).unwrap());
            }
        }
        for b in &self.blacks {
            for (i, cell) in b.cells.iter().enumerate() {
                if i > 0 {
                    let prev = grid.get(b.cells[i-1]).0.clone();
                    if i < b.cells.len() - 1 {
                        let next = grid.get(b.cells[i+1]).0.clone();
                        grid.get_mut(*cell).0.intersect_with(
                            &kropki_black_between::<MIN, MAX, _>(&prev, &Some(next), b.mutually_visible)
                        );
                    } else {
                        grid.get_mut(*cell).0.intersect_with(
                            &kropki_black_between::<MIN, MAX, _>(&prev, &None, b.mutually_visible)
                        );
                    }
                    if !kropki_black_adj_ok::<MIN, MAX, _>(&prev, &grid.get(*cell).0) {
                        return ConstraintResult::Contradiction(self.kb_if_attr.clone());
                    }
                } else {
                    // Note: guaranteed that i < b.cells.len() - 1
                    let next = grid.get(b.cells[i+1]).0.clone();
                    grid.get_mut(*cell).0.intersect_with(
                        &kropki_black_between::<MIN, MAX, _>(&next, &None, b.mutually_visible)
                    );
                }
            }
        }
        for w in &self.whites {
            for (i, cell) in w.cells.iter().enumerate() {
                if i > 0 {
                    let prev = grid.get(w.cells[i-1]).0.clone();
                    if i < w.cells.len() - 1 {
                        let next = grid.get(w.cells[i+1]).0.clone();
                        grid.get_mut(*cell).0.intersect_with(
                            &kropki_white_between::<MIN, MAX, _>(&prev, &Some(next), w.mutually_visible)
                        );
                    } else {
                        grid.get_mut(*cell).0.intersect_with(
                            &kropki_white_between::<MIN, MAX, _>(&prev, &None, w.mutually_visible)
                        );
                    }
                    if !kropki_white_adj_ok::<MIN, MAX, _>(&prev, &grid.get(*cell).0) {
                        return ConstraintResult::Contradiction(self.kw_if_attr.clone());
                    }
                } else {
                    // Note: guaranteed that i < w.cells.len() - 1
                    let next = grid.get(w.cells[i+1]).0.clone();
                    grid.get_mut(*cell).0.intersect_with(
                        &kropki_white_between::<MIN, MAX, _>(&next, &None, w.mutually_visible)
                    );
                }
            }
        }
        ConstraintResult::Ok
    }

    fn debug_at(&self, _: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, index: Index) -> Option<String> {
        let header = "KropkiChecker:\n";
        let mut lines = vec![];
        if let Some(s) = self.illegal.debug_at(index) {
            lines.push(format!("  {}", s));
        }
        for (i, b) in self.blacks.iter().enumerate() {
            if !b.contains(index) {
                continue;
            }
            lines.push(format!("  Black[{}]: ", i));
            for cell in &b.cells {
                if *cell != index {
                    continue;
                }
                let rem = self.black_remaining.get(cell)
                    .expect(format!("remaining[{:?}] not found!", cell).as_str());
                lines.push(format!("  - remaining vals: {:?}", unpack_stdval_vals::<MIN, MAX, _>(rem)));
            }
        }
        for (i, w) in self.whites.iter().enumerate() {
            if !w.contains(index) {
                continue;
            }
            lines.push(format!("  White[{}]: ", i));
            for cell in &w.cells {
                if *cell != index {
                    continue;
                }
                let rem = self.white_remaining.get(cell)
                    .expect(format!("remaining[{:?}] not found!", cell).as_str());
                lines.push(format!("  - remaining vals: {:?}", unpack_stdval_vals::<MIN, MAX, _>(rem)));
            }
        }
        if lines.is_empty() {
            None
        } else {
            Some(format!("{}{}", header, lines.join("\n")))
        }
    }

    fn debug_highlight(&self, _: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, index: Index) -> Option<(u8, u8, u8)> {
        if let Some(c) = self.illegal.debug_highlight(index) {
            return Some(c);
        }
        let mut b_color = None;
        for (i, b) in self.blacks.iter().enumerate() {
            if b.contains(index) {
                b_color = Some(self.black_colors[i]);
                break;
            }
        }
        let mut w_color = None;
        for (i, w) in self.whites.iter().enumerate() {
            if w.contains(index) {
                w_color = Some(self.white_colors[i]);
                break;
            }
        }
        color_opt_ave2(b_color, w_color)
    }
}

#[cfg(test)]
mod test {
    use crate::{constraint::{test_util::{assert_contradiction, assert_no_contradiction}, MultiConstraint}, ranker::StdRanker, solver::test_util::PuzzleReplay, sudoku::{four_standard_parse, six_standard_overlay, unpack_stdval_vals, SixStdVal, StdChecker}};
    use super::*;

    fn assert_black_possible<const MIN: u8, const MAX: u8>(
        expected: Vec<u8>,
    ) {
        assert_eq!(
            unpack_stdval_vals::<MIN, MAX, _>(&kropki_black_possible::<MIN, MAX>().get()),
            expected,
            "Possible vals for black kropkis w/{}..={} should be {:?}",
            MIN, MAX, expected,
        );
    }

    #[test]
    fn test_kropki_black_possible() {
        assert_black_possible::<1, 9>(
            vec![1, 2, 3, 4, 6, 8],
        );
        assert_black_possible::<1, 6>(
            vec![1, 2, 3, 4, 6],
        );
        assert_black_possible::<5, 20>(
            vec![5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],
        );
    }

    fn assert_black_chain_possible<const MIN: u8, const MAX: u8>(
        chain_len: usize,
        len_from_chain_end: usize,
        expected: Vec<u8>,
    ) {
        assert_eq!(
            unpack_stdval_vals::<MIN, MAX, _>(&kropki_black_possible_chain::<MIN, MAX>().get(
                chain_len,
                len_from_chain_end,
            )),
            expected,
            "Possible vals for position {} in black kropki chains of len {} w/{}..={} should be {:?}",
            len_from_chain_end, chain_len, MIN, MAX, expected,
        );
    }

    #[test]
    fn test_kropki_black_possible_chain() {
        // Same as being possible at all
        assert_black_chain_possible::<1, 9>(
            2, 0,
            vec![1, 2, 3, 4, 6, 8],
        );
        // Just demonstrating that this works when the distance from the
        // end is past the halfway mark.
        assert_black_chain_possible::<1, 9>(
            2, 1,
            vec![1, 2, 3, 4, 6, 8],
        );
        // 1-2-4-8 is the only chain longer than 2 in [1,9]
        assert_black_chain_possible::<1, 9>(
            3, 0,
            vec![1, 2, 4, 8],
        );
        // Only 2-4 works in the middle position
        assert_black_chain_possible::<1, 9>(
            3, 1,
            vec![2, 4],
        );
        // With length of 4, it must be 1-2-4-8 or 8-4-2-1
        assert_black_chain_possible::<1, 9>(
            4, 0,
            vec![1, 8],
        );
        assert_black_chain_possible::<1, 9>(
            4, 1,
            vec![2, 4],
        );
        assert_black_chain_possible::<1, 9>(
            5, 0,
            vec![],
        );
        // Same as being possible at all
        assert_black_chain_possible::<1, 6>(
            2, 0,
            vec![1, 2, 3, 4, 6],
        );
        // 1-2-4 is the only chain longer than 2 in [1,6]
        assert_black_chain_possible::<1, 6>(
            3, 0,
            vec![1, 4],
        );
        assert_black_chain_possible::<1, 6>(
            3, 1,
            vec![2],
        );
        assert_black_chain_possible::<1, 6>(
            4, 0,
            vec![],
        );
        // Same as being possible at all
        assert_black_chain_possible::<5, 20>(
            2, 0,
            vec![5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],
        );
        // 5-10-20 is the only chain longer than 2 in [5,20]
        assert_black_chain_possible::<5, 20>(
            3, 0,
            vec![5, 20],
        );
        assert_black_chain_possible::<5, 20>(
            3, 1,
            vec![10],
        );
        assert_black_chain_possible::<5, 20>(
            4, 0,
            vec![],
        );
    }

    fn assert_black_adj_ok<const MIN: u8, const MAX: u8>(
        a: Vec<u8>, b: Vec<u8>, expected: bool,
    ) {
        let a_set = VBitSet::<StdVal<MIN, MAX>>::from_values(
            &a.iter().map(|v| StdVal::new(*v)).collect()
        );
        let b_set = VBitSet::<StdVal<MIN, MAX>>::from_values(
            &b.iter().map(|v| StdVal::new(*v)).collect()
        );
        if expected {
            assert!(
                kropki_black_adj_ok::<MIN, MAX, _>(&a_set, &b_set),
                "Expected {:?} and {:?} to be ok adjacent on a black kropki",
                a, b
            );
        } else {
            assert!(
                !kropki_black_adj_ok::<MIN, MAX, _>(&a_set, &b_set),
                "Expected {:?} and {:?} not to be ok adjacent on a black kropki",
                a, b
            );
        }
    }

    #[test]
    fn test_kropki_black_adj_ok() {
        assert_black_adj_ok::<1, 9>(
            vec![1],
            vec![2, 4, 8],
            true,
        );
        assert_black_adj_ok::<1, 9>(
            vec![1],
            vec![3, 4, 6, 8],
            false,
        );
        assert_black_adj_ok::<1, 9>(
            vec![1, 2, 4, 8],
            vec![3, 6],
            false,
        );
        assert_black_adj_ok::<1, 9>(
            vec![2, 4],
            vec![1, 8],
            true,
        );
    }

    fn assert_black_between<const MIN: u8, const MAX: u8>(
        left: Vec<u8>, right: Option<Vec<u8>>, mutually_visible: bool, expected: Vec<u8>,
    ) {
        let left_set = VBitSet::<StdVal<MIN, MAX>>::from_values(
            &left.iter().map(|v| StdVal::new(*v)).collect()
        );
        let right_copy = right.clone();
        let right_set = right.map(|r| {
            VBitSet::<StdVal<MIN, MAX>>::from_values(
                &r.iter().map(|v| StdVal::new(*v)).collect()
            )
        });
        assert_eq!(
            unpack_stdval_vals::<MIN, MAX, _>(&kropki_black_between::<MIN, MAX, _>(
                &left_set, &right_set, mutually_visible,
            )),
            expected,
            "Expected valid (black) values between {:?} and {:?} (mutually \
             visible: {}) to be {:?}",
            left, right_copy, mutually_visible, expected
        );
    }

    #[test]
    fn test_kropki_black_between() {
        assert_black_between::<1, 9>(
            vec![1],
            Some(vec![2, 4, 8]),
            true,
            vec![2],
        );
        assert_black_between::<1, 9>(
            vec![2, 8],
            Some(vec![2, 8]),
            true,
            vec![4],
        );
        assert_black_between::<1, 9>(
            vec![2, 8],
            Some(vec![2, 8]),
            false,
            vec![1, 4],
        );
        assert_black_between::<1, 9>(
            vec![3],
            Some(vec![3]),
            false,
            vec![6],
        );
        assert_black_between::<1, 9>(
            vec![3],
            Some(vec![3]),
            true,
            vec![],
        );
        assert_black_between::<1, 9>(
            vec![1],
            None,
            true,
            vec![2],
        );
        assert_black_between::<1, 9>(
            vec![6],
            None,
            true,
            vec![3],
        );
        assert_black_between::<1, 9>(
            vec![2],
            None,
            true,
            vec![1, 4],
        );
    }

    fn assert_white_chain_possible<const MIN: u8, const MAX: u8>(
        chain_len: usize,
        len_from_chain_end: usize,
        expected: Vec<u8>,
    ) {
        assert_eq!(
            unpack_stdval_vals::<MIN, MAX, _>(&kropki_white_possible_chain::<MIN, MAX>().get(
                chain_len,
                len_from_chain_end,
            )),
            expected,
            "Possible vals for position {} in white kropki chains of len {} w/{}..={} should be {:?}",
            len_from_chain_end, chain_len, MIN, MAX, expected,
        );
    }

    #[test]
    fn test_kropki_white_possible_chain() {
        // Same as being possible at all
        assert_white_chain_possible::<1, 9>(
            2, 0,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
        );
        // Just demonstrating that this works when the distance from the
        // end is past the halfway mark.
        assert_white_chain_possible::<1, 9>(
            2, 1,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
        );
        // 1..=5 is the only chain of length five in [1,9]
        // Only 1-5 at the ends
        assert_white_chain_possible::<1, 5>(
            5, 0,
            vec![1, 5],
        );
        // Only 2-4 and the next position in
        assert_white_chain_possible::<1, 5>(
            5, 1,
            vec![2, 4],
        );
        // Only 3 in the middle
        assert_white_chain_possible::<1, 5>(
            5, 2,
            vec![3],
        );
        // With length of 4, it must be either 1..=4 or 2..=5
        assert_white_chain_possible::<1, 5>(
            4, 0,
            vec![1, 2, 4, 5],
        );
        assert_white_chain_possible::<1, 5>(
            4, 1,
            vec![2, 3, 4],
        );
    }

    fn assert_white_adj_ok<const MIN: u8, const MAX: u8>(
        a: Vec<u8>, b: Vec<u8>, expected: bool,
    ) {
        let a_set = VBitSet::<StdVal<MIN, MAX>>::from_values(
            &a.iter().map(|v| StdVal::new(*v)).collect()
        );
        let b_set = VBitSet::<StdVal<MIN, MAX>>::from_values(
            &b.iter().map(|v| StdVal::new(*v)).collect()
        );
        if expected {
            assert!(
                kropki_white_adj_ok::<MIN, MAX, _>(&a_set, &b_set),
                "Expected {:?} and {:?} to be ok adjacent on a white kropki",
                a, b
            );
        } else {
            assert!(
                !kropki_white_adj_ok::<MIN, MAX, _>(&a_set, &b_set),
                "Expected {:?} and {:?} not to be ok adjacent on a white kropki",
                a, b
            );
        }
    }

    #[test]
    fn test_kropki_white_adj_ok() {
        assert_white_adj_ok::<1, 9>(
            vec![1],
            vec![2, 3],
            true,
        );
        assert_white_adj_ok::<1, 9>(
            vec![1, 7],
            vec![3, 4, 5],
            false,
        );
        assert_white_adj_ok::<1, 9>(
            vec![2, 3, 4],
            vec![6, 7, 8],
            false,
        );
        assert_white_adj_ok::<1, 9>(
            vec![2, 4],
            vec![1, 7],
            true,
        );
    }

    fn assert_white_between<const MIN: u8, const MAX: u8>(
        left: Vec<u8>, right: Option<Vec<u8>>, mutually_visible: bool, expected: Vec<u8>,
    ) {
        let left_set = VBitSet::<StdVal<MIN, MAX>>::from_values(
            &left.iter().map(|v| StdVal::new(*v)).collect()
        );
        let right_copy = right.clone();
        let right_set = right.map(|r| {
            VBitSet::<StdVal<MIN, MAX>>::from_values(
            &r.iter().map(|v| StdVal::new(*v)).collect()
            )
        });
        assert_eq!(
            unpack_stdval_vals::<MIN, MAX, _>(&kropki_white_between::<MIN, MAX, _>(
                &left_set, &right_set, mutually_visible,
            )),
            expected,
            "Expected valid (white) values between {:?} and {:?} (mutually \
             visible: {}) to be {:?}",
            left, right_copy, mutually_visible, expected
        );
    }

    #[test]
    fn test_kropki_white_between() {
        assert_white_between::<1, 9>(
            vec![1, 7],
            Some(vec![3, 5, 9]),
            true,
            vec![2, 6, 8],
        );
        assert_white_between::<1, 9>(
            vec![1, 3],
            Some(vec![1, 3]),
            true,
            vec![2],
        );
        assert_white_between::<1, 9>(
            vec![1, 3],
            Some(vec![1, 3]),
            false,
            vec![2, 4],
        );
        assert_white_between::<1, 9>(
            vec![3],
            Some(vec![3]),
            false,
            vec![2, 4],
        );
        assert_white_between::<1, 9>(
            vec![3],
            Some(vec![3]),
            true,
            vec![],
        );
        assert_white_between::<1, 9>(
            vec![1],
            None,
            true,
            vec![2],
        );
        assert_white_between::<1, 9>(
            vec![3],
            None,
            true,
            vec![2, 4],
        );
    }

    // This is a 6x6 puzzle with a black kropki chain from [1, 0] to [1, 2] and
    // a black kropki dot from [1, 4] to [1, 5]. Call with different givens and
    // an expectation for it to return a contradiction (or not).
    fn assert_kropki_black_result(
        setup: &str, 
        expected: Option<&'static str>,
    ) {
        let overlay = six_standard_overlay();
        let kb = KropkiBuilder::new(&overlay);
        let kropkis = vec![
            kb.b_polyline(vec![[1, 0], [1, 2]]),
            kb.b_across([1, 4]),
        ];
        let ranker = StdRanker::default();
        let mut puzzle = overlay.parse_state::<SixStdVal>(setup).unwrap();
        let mut constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(&overlay),
            KropkiChecker::new(kropkis),
        ]);
        let result = PuzzleReplay::new(&mut puzzle, &ranker, &mut constraint, None).replay().unwrap();
        if let Some(attr) = expected {
            assert_contradiction(result, attr);
        } else {
            assert_no_contradiction(result);
        }
    }

    #[test]
    fn test_kropki_black_1_not_middle() {
        // 1 can't be a middle value at all
        let input: &str = ". . .|. . .\n\
                           . 1 .|. . .\n\
                           -----+-----\n\
                           . . .|. . .\n\
                           . . .|. . .\n\
                           -----+-----\n\
                           . . .|. . .\n\
                           . . .|. . .\n";
        assert_kropki_black_result(input, Some("KROPKI_BLACK_CONFLICT"));
    }

    #[test]
    fn test_kropki_black_sudoku_interaction() {
        // [1, 1] has to be a 2 for KB reasons but is ruled out for sudoku ones
        let input: &str = ". . .|. . .\n\
                           4 . .|. 2 .\n\
                           -----+-----\n\
                           . . .|. . .\n\
                           . . .|. . .\n\
                           -----+-----\n\
                           . . .|. . .\n\
                           . . .|. . .\n";
        assert_kropki_black_result(input, Some("KROPKI_BLACK_INFEASIBLE"));
    }

    #[test]
    fn test_kropki_black_contradiction() {
        // Straightforward contradiction
        let input: &str = ". . .|. . .\n\
                           4 1 .|. . .\n\
                           -----+-----\n\
                           . . .|. . .\n\
                           . . .|. . .\n\
                           -----+-----\n\
                           . . .|. . .\n\
                           . . .|. . .\n";
        assert_kropki_black_result(input, Some("KROPKI_BLACK_CONFLICT"));
    }

    #[test]
    fn test_kropki_black_valid_fill() {
        // Valid fill
        let input: &str = ". . .|. . .\n\
                           1 2 4|5 3 6\n\
                           -----+-----\n\
                           . . .|. . .\n\
                           . . .|. . .\n\
                           -----+-----\n\
                           . . .|. . .\n\
                           . . .|. . .\n";
        assert_kropki_black_result(input, None);
    }

    // This is a 4x4 puzzle with a white kropki chain from [0, 0] to [0, 2] and
    // a white kropki dot from [1, 0] to [2, 0]. Call with different givens and
    // an expectation for it to return a contradiction (or not).
    fn assert_kropki_white_result(
        setup: &str, 
        expected: Option<&'static str>,
    ) {
        let mut puzzle = four_standard_parse(setup).unwrap();
        let kb = KropkiBuilder::new(puzzle.overlay());
        let kropkis = vec![
            kb.w_polyline(vec![[0, 0], [0, 2]]),
            kb.w_down([1, 0]),
        ];
        let ranker = StdRanker::default();
        let mut constraint = MultiConstraint::new(vec_box::vec_box![
            StdChecker::new(puzzle.overlay()),
            KropkiChecker::new(kropkis),
        ]);
        let result = PuzzleReplay::new(&mut puzzle, &ranker, &mut constraint, None).replay().unwrap();
        if let Some(attr) = expected {
            assert_contradiction(result, attr);
        } else {
            assert_no_contradiction(result);
        }
    }

    #[test]
    fn test_kropki_white_1_not_middle() {
        // 1 can't be a middle value at all
        let input: &str = ". 1|. .\n\
                           . .|. .\n\
                           ---+---\n\
                           . .|. .\n\
                           . .|. .\n";
        assert_kropki_white_result(input, Some("KROPKI_WHITE_CONFLICT"));
    }

    #[test]
    fn test_kropki_white_sudoku_interaction() {
        // [0, 1] has to be a 2 for KB reasons but is ruled out for sudoku ones
        let input: &str = "1 .|. .\n\
                           . 2|. .\n\
                           ---+---\n\
                           . .|. .\n\
                           . .|. .\n";
        assert_kropki_white_result(input, Some("KROPKI_WHITE_INFEASIBLE"));
    }

    #[test]
    fn test_kropki_white_contradiction() {
        // Direct contradiction
        let input: &str = "1 3|. .\n\
                           . .|. .\n\
                           ---+---\n\
                           . .|. .\n\
                           . .|. .\n";
        assert_kropki_white_result(input, Some("KROPKI_WHITE_INFEASIBLE"));
    }

    #[test]
    fn test_kropki_white_valid_fill() {
        // Valid fill
        let input: &str = "1 2|3 4\n\
                           4 .|. .\n\
                           ---+---\n\
                           3 .|. .\n\
                           . .|. .\n";
        assert_kropki_white_result(input, None);
    }
}