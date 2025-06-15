use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::{LazyLock, Mutex};
use crate::constraint::{Constraint, ConstraintViolationDetail};
use crate::core::{empty_set, singleton_set, ConstraintResult, DecisionGrid, FKWithId, FeatureKey, Index, State, Stateful, UVSet, Value};
use crate::sudoku::{unpack_sval_vals, Overlay, SState, SVal};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum KropkiColor { Black, White }

#[derive(Debug, Clone)]
pub struct KropkiDotChain {
    color: KropkiColor,
    cells: Vec<Index>,
    mutually_visible: bool,
}

pub struct KropkiBuilder<'a, O: Overlay>(&'a O);

impl <'a, O: Overlay> KropkiBuilder<'a, O> {
    pub fn new(overlay: &'a O) -> Self { Self(overlay) }

    fn create(&self, color: KropkiColor, cells: Vec<Index>) -> KropkiDotChain {
        for (i, &cell) in cells.iter().enumerate() {
            if i > 0 {
                let prev = cells[i - 1];
                let diff = (cell[0].abs_diff(prev[0]), cell[1].abs_diff(prev[1]));
                if diff != (0, 1) && diff != (1, 0) && diff != (1, 1) {
                    panic!("Cells {:?} and {:?} are not adjacent", cell, prev);
                }
            }
            if i < cells.len() - 1{
                let next = cells[i + 1];
                let diff = (cell[0].abs_diff(next[0]), cell[1].abs_diff(next[1]));
                if diff != (0, 1) && diff != (1, 0) && diff != (1, 1) {
                    panic!("Cells {:?} and {:?} are not adjacent", cell, next);
                }
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
}

/// Tables of useful sets for kropki black dot constraints.
static KB_POSSIBLE: LazyLock<Mutex<HashMap<(u8, u8), UVSet<u8>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});
static KB_POSSIBLE_MV: LazyLock<Mutex<HashMap<(u8, u8, usize), Vec<UVSet<u8>>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

fn kropki_black_possible<const MIN: u8, const MAX: u8>() -> UVSet<u8> {
    let mut map = KB_POSSIBLE.lock().unwrap();
    map.entry((MIN, MAX)).or_insert_with(|| {
        let mut set = empty_set::<u8, SVal<MIN, MAX>>();
        for v in SVal::<MIN, MAX>::possibilities() {
            if v.val() % 2 == 0 {
                let half = v.val() / 2;
                if MIN <= half {
                    set.insert(v.to_uval());
                    continue;
                }
            }
            if v.val() < 128 {
                let double = v.val() * 2;
                if double <= MAX {
                    set.insert(v.to_uval());
                    continue;
                }
            }
        }
        set
    }).clone()
}

fn kropki_black_possible_chain<const MIN: u8, const MAX: u8>(n_mutually_visible: usize, mut len_from_end: usize) -> UVSet<u8> {
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
    let mut map = KB_POSSIBLE_MV.lock().unwrap();
    let e = map.entry((MIN, MAX, n_mutually_visible)).or_insert_with(|| {
        let mut possible = vec![empty_set::<u8, SVal<MIN, MAX>>(); n_mutually_visible/2 + n_mutually_visible % 2];
        for v in SVal::<MIN, MAX>::possibilities() {
            let mut chain: Vec<SVal<MIN, MAX>> = vec![];
            let mut cur = v.val() as u16;
            while cur <= (MAX as u16) && chain.len() < n_mutually_visible {
                chain.push(SVal::new(cur as u8));
                cur *= 2;
            }
            if chain.len() != n_mutually_visible {
                continue;
            }
            for i in 0..possible.len() {
                possible[i].insert(chain[i].to_uval());
                possible[i].insert(chain[chain.len()-1-i].to_uval());
            }
        }
        possible
    });
    e[len_from_end].clone()
}

fn kropki_black_adj_ok<const MIN: u8, const MAX: u8>(a: &UVSet<u8>, b: &UVSet<u8>) -> bool {
    for v1 in unpack_sval_vals::<MIN, MAX>(a) {
        for v2 in unpack_sval_vals::<MIN, MAX>(b) {
            if (v1 < v2 && v1 < 128 && v1*2 == v2) ||
               (v2 < v1 && v2 < 128 && v2*2 == v1) {
                return true;
            }
        }
    }
    false
}

fn get_lower<const MIN: u8, const MAX: u8>(v: SVal<MIN, MAX>) -> Option<SVal<MIN, MAX>> {
    if v.val() % 2 == 0 && v.val()/2 >= MIN {
        Some(SVal::new(v.val()/2))
    } else {
        None
    }
}

fn get_upper<const MIN: u8, const MAX: u8>(v: SVal<MIN, MAX>) -> Option<SVal<MIN, MAX>> {
    if v.val() < 128 && v.val()*2 <= MAX {
        Some(SVal::new(v.val()*2))
    } else {
        None
    }
}

fn kropki_black_between<const MIN: u8, const MAX: u8>(left: &UVSet<u8>, right: &UVSet<u8>, mutually_visible: bool) -> UVSet<u8> {
    let mut possible = empty_set::<u8, SVal<MIN, MAX>>();
    for v in SVal::<MIN, MAX>::possibilities() {
        let (lower, upper) = (get_lower::<MIN, MAX>(v), get_upper::<MIN, MAX>(v));
        if mutually_visible {
            if lower.is_none() || upper.is_none() {
                continue;
            }
            let (l, u) = (lower.unwrap(), upper.unwrap());
            if (left.contains(l.to_uval()) && right.contains(u.to_uval())) ||
               (left.contains(u.to_uval()) && right.contains(l.to_uval())) {
                possible.insert(v.to_uval());
            }
        } else if lower.is_some() && upper.is_some() {
            let (l, u) = (lower.unwrap(), upper.unwrap());
            if (left.contains(l.to_uval()) || left.contains(u.to_uval())) &&
               (right.contains(l.to_uval()) || right.contains(u.to_uval())) {
                possible.insert(v.to_uval());
            }
        } else if let Some(l) = lower {
            if left.contains(l.to_uval()) && right.contains(l.to_uval()) {
                possible.insert(v.to_uval());
            }
        } else if let Some(u) = upper {
            if left.contains(u.to_uval()) && right.contains(u.to_uval()) {
                possible.insert(v.to_uval());
            }
        }
    }
    possible
}

// TODO: Useful functions for kropki white dots

pub const KROPKI_BLACK_FEATURE: &str = "KROPKI_BLACK";

// TODO: Add support for white kropki dots
pub struct KropkiChecker<const MIN: u8, const MAX: u8> {
    blacks: Vec<KropkiDotChain>,
    black_remaining: HashMap<Index, UVSet<u8>>,
    kb_feature: FeatureKey<FKWithId>,
}

impl <const MIN: u8, const MAX: u8> KropkiChecker<MIN, MAX> {
    pub fn new(chains: Vec<KropkiDotChain>) -> Self {
        if chains.iter().any(|c| c.color == KropkiColor::White) {
            panic!("White kropki dot support not yet implemented!");
        }
        let mut covered = HashSet::new();
        for b in &chains {
            for cell in &b.cells {
                if covered.contains(&cell) {
                    panic!("Multiple black kropki chains contain cell: {:?}\n", cell);
                }
                covered.insert(cell);
            }
        }
        let mut kc = Self {
            blacks: chains,
            black_remaining: HashMap::new(),
            kb_feature: FeatureKey::new(KROPKI_BLACK_FEATURE).unwrap(),
        };
        kc.reset();
        kc
    }
}

impl <const MIN: u8, const MAX: u8> Debug for KropkiChecker<MIN, MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, b) in self.blacks.iter().enumerate() {
            write!(f, " Black[{}]: ", i)?;
            for cell in &b.cells {
                let rem = self.black_remaining.get(cell)
                    .expect(format!("remaining[{:?}] not found!", cell).as_str());
                write!(f, "{:?}=>{:?} ", cell, unpack_sval_vals::<MIN, MAX>(rem))?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8> Stateful<u8, SVal<MIN, MAX>> for KropkiChecker<MIN, MAX> {
    fn reset(&mut self) {
        for b in self.blacks.iter() {
            if b.mutually_visible {
                for (i, cell) in b.cells.iter().enumerate() {
                    self.black_remaining.insert(
                        *cell,
                        kropki_black_possible_chain::<MIN, MAX>(b.cells.len(), i),
                    );
                }
            } else {
                for cell in &b.cells {
                    self.black_remaining.insert(
                        *cell, 
                        kropki_black_possible::<MIN, MAX>(),
                    );
                }
            }
        }
    }

    fn apply(&mut self, index: Index, value: SVal<MIN, MAX>) -> Result<(), crate::core::Error> {
        if let Some(r) = self.black_remaining.get_mut(&index) {
            *r = singleton_set::<u8, SVal<MIN, MAX>>(value);
        }
        Ok(())
    }

    fn undo(&mut self, index: Index, _: SVal<MIN, MAX>) -> Result<(), crate::core::Error> {
        if !self.black_remaining.contains_key(&index) {
            return Ok(());
        }
        for b in &self.blacks {
            for (i, cell) in b.cells.iter().enumerate() {
                if *cell != index {
                    continue;
                }
                *self.black_remaining.get_mut(cell).unwrap() = if b.mutually_visible {
                    kropki_black_possible_chain::<MIN, MAX>(b.cells.len(), i)
                } else {
                    kropki_black_possible::<MIN, MAX>()
                };
                break;
            }
        }
        Ok(())
    }
}

impl <const N: usize, const M: usize, const MIN: u8, const MAX: u8, O: Overlay>
Constraint<u8, SState<N, M, MIN, MAX, O>> for KropkiChecker<MIN, MAX> {
    fn check(&self, puzzle: &SState<N, M, MIN, MAX, O>, grid: &mut DecisionGrid<u8, SVal<MIN, MAX>>) -> ConstraintResult<u8, SVal<MIN, MAX>> {
        for b in &self.blacks {
            for cell in &b.cells {
                if puzzle.get(*cell).is_some() {
                    continue;
                }
                let g = grid.get_mut(*cell);
                if puzzle.get(*cell).is_none() {
                    g.0.intersect_with(&self.black_remaining.get(cell).unwrap());
                }
                g.1.add(&self.kb_feature, 1.0);
            }
        }
        for b in &self.blacks {
            for (i, cell) in b.cells.iter().enumerate() {
                if puzzle.get(*cell).is_some() {
                    continue;
                }
                if i > 0 {
                    let prev = grid.get(b.cells[i-1]).0.clone();
                    if i < b.cells.len() - 1 {
                        let next = grid.get(b.cells[i+1]).0.clone();
                        grid.get_mut(*cell).0.intersect_with(
                            &kropki_black_between::<MIN, MAX>(&prev, &next, b.mutually_visible)
                        );
                    }
                    if !kropki_black_adj_ok::<MIN, MAX>(&prev, &grid.get(*cell).0) {
                        return ConstraintResult::Contradiction;
                    }
                }
            }
        }
        ConstraintResult::Ok
    }

    fn explain_contradictions(&self, _: &SState<N, M, MIN, MAX, O>) -> Vec<ConstraintViolationDetail> {
        todo!()
    }
}


#[cfg(test)]
mod test {
    use crate::{core::pack_values, sudoku::unpack_sval_vals};
    use super::*;

    fn assert_black_possible<const MIN: u8, const MAX: u8>(
        expected: Vec<u8>,
    ) {
        assert_eq!(
            unpack_sval_vals::<MIN, MAX>(&kropki_black_possible::<MIN, MAX>()),
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
            unpack_sval_vals::<MIN, MAX>(&kropki_black_possible_chain::<MIN, MAX>(
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
        let a_set = pack_values::<u8, SVal<MIN, MAX>>(
            &a.iter().map(|v| SVal::new(*v)).collect()
        );
        let b_set = pack_values::<u8, SVal<MIN, MAX>>(
            &b.iter().map(|v| SVal::new(*v)).collect()
        );
        if expected {
            assert!(
                kropki_black_adj_ok::<MIN, MAX>(&a_set, &b_set),
                "Expected {:?} and {:?} to be ok adjacent on a black kropki",
                a, b
            );
        } else {
            assert!(
                !kropki_black_adj_ok::<MIN, MAX>(&a_set, &b_set),
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
        left: Vec<u8>, right: Vec<u8>, mutually_visible: bool, expected: Vec<u8>,
    ) {
        let left_set = pack_values::<u8, SVal<MIN, MAX>>(
            &left.iter().map(|v| SVal::new(*v)).collect()
        );
        let right_set = pack_values::<u8, SVal<MIN, MAX>>(
            &right.iter().map(|v| SVal::new(*v)).collect()
        );
        assert_eq!(
            unpack_sval_vals::<MIN, MAX>(&kropki_black_between::<MIN, MAX>(
                &left_set, &right_set, mutually_visible,
            )),
            expected,
            "Expected valid values between {:?} and {:?} (mutually visible: {}) \
             to be {:?}",
            left, right, mutually_visible, expected
        );
    }

    #[test]
    fn test_kropki_black_between() {
        assert_black_between::<1, 9>(
            vec![1],
            vec![2, 4, 8],
            true,
            vec![2],
        );
        assert_black_between::<1, 9>(
            vec![2, 8],
            vec![2, 8],
            true,
            vec![4],
        );
        assert_black_between::<1, 9>(
            vec![2, 8],
            vec![2, 8],
            false,
            vec![1, 4],
        );
        assert_black_between::<1, 9>(
            vec![3],
            vec![3],
            false,
            vec![6],
        );
        assert_black_between::<1, 9>(
            vec![3],
            vec![3],
            true,
            vec![],
        );
    }
    
    // TODO: test the constraint
}