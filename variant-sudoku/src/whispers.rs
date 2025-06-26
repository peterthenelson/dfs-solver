use std::{collections::HashMap, sync::{LazyLock, Mutex}};
use crate::{core::{VBitSet, VBitSetRef, VDenseMap, VDenseMapRef, VMap, VMapMut, VSet, Value}, memo::{FnToCalc, MemoLock}, sudoku::StdVal};

/// Tables of useful sets for whisper-style constraints.
static WHISPER_NEIGHBORS: LazyLock<Mutex<HashMap<(u8, u8, u8), Box<[bit_set::BitSet]>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});
static WHISPER_POSSIBLE_VALS: LazyLock<Mutex<HashMap<(u8, u8, u8, bool), bit_set::BitSet>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

fn whisper_neighbors_raw<const MIN: u8, const MAX: u8>(args: &(u8,)) -> Box<[bit_set::BitSet]> {
    let dist = args.0;
    let mut neighbors = VDenseMap::<StdVal<MIN, MAX>, VBitSet<StdVal<MIN, MAX>>>::filled(
        VBitSet::<StdVal<MIN, MAX>>::empty()
    );
    for v1 in StdVal::<MIN, MAX>::possibilities() {
        for v2 in StdVal::<MIN, MAX>::possibilities() {
            if v1 == v2 {
                continue;
            }
            let d = v1.val().abs_diff(v2.val());
            if d >= dist {
                neighbors.get_mut(&v1).insert(&v2);
            }
        }
    }
    let mut partial = VDenseMap::<StdVal<MIN, MAX>, bit_set::BitSet>::empty();
    for (k, v) in neighbors.iter() {
        *partial.get_mut(&k) = v.into_erased();
    }
    partial.into_erased()
}

pub struct WhisperNeighbors<const MIN: u8, const MAX: u8>(MemoLock<(u8,), (u8, u8, u8), Box<[bit_set::BitSet]>, FnToCalc<(u8,), (u8, u8, u8), Box<[bit_set::BitSet]>>>);
impl <const MIN: u8, const MAX: u8> WhisperNeighbors<MIN, MAX> {
    pub fn get(&mut self, dist: u8, val: StdVal<MIN, MAX>) -> VBitSetRef<StdVal<MIN, MAX>> {
        let map = VDenseMapRef::<StdVal<MIN, MAX>, bit_set::BitSet>::assume_typed(
            self.0.get(&(dist,)),
        );
        VBitSetRef::<StdVal<MIN, MAX>>::assume_typed(
            map.get_into(&val)
        )
    }
}

pub fn whisper_neighbors<const MIN: u8, const MAX: u8>() -> WhisperNeighbors<MIN, MAX> {
    let guard = WHISPER_NEIGHBORS.lock().unwrap();
    let calc = FnToCalc::<_, _, _>::new(
        |&(dist,)| (MIN, MAX, dist),
        whisper_neighbors_raw::<MIN, MAX>,
    );
    WhisperNeighbors(MemoLock::new(guard, calc))
}

fn whisper_possible_vals_raw<const MIN: u8, const MAX: u8>(args: &(u8, bool)) -> bit_set::BitSet {
    let dist = args.0;
    let has_two_mutually_visible_neighbors = args.1;
    let mut possible_vals = VBitSet::<StdVal<MIN, MAX>>::empty();
    for v in StdVal::<MIN, MAX>::possibilities() {
        let mut wn = whisper_neighbors::<MIN, MAX>();
        let neighbors = wn.get(dist, v);
        if neighbors.len() >= 2 || (!neighbors.is_empty() && !has_two_mutually_visible_neighbors) {
            possible_vals.insert(&v);
        }
    }
    possible_vals.into_erased()
}

pub struct WhisperPossibleValues<const MIN: u8, const MAX: u8>(MemoLock<(u8, bool), (u8, u8, u8, bool), bit_set::BitSet, FnToCalc<(u8, bool), (u8, u8, u8, bool), bit_set::BitSet>>);
impl <const MIN: u8, const MAX: u8> WhisperPossibleValues<MIN, MAX> {
    pub fn get(&mut self, dist: u8, has_two_mutually_visible_neighbors: bool) -> VBitSetRef<StdVal<MIN, MAX>> {
        VBitSetRef::<StdVal<MIN, MAX>>::assume_typed(
            self.0.get(&(dist, has_two_mutually_visible_neighbors)),
        )
    }
}

pub fn whisper_possible_values<const MIN: u8, const MAX: u8>() -> WhisperPossibleValues<MIN, MAX> {
    let guard = WHISPER_POSSIBLE_VALS.lock().unwrap();
    let calc = FnToCalc::<_, _, _>::new(
        |&(dist, h2mvn)| (MIN, MAX, dist, h2mvn),
        whisper_possible_vals_raw::<MIN, MAX>,
    );
    WhisperPossibleValues(MemoLock::new(guard, calc))
}

pub fn whisper_between<const MIN: u8, const MAX: u8, VS: VSet<StdVal<MIN, MAX>>>(dist: u8, left: &VS, right: &VS) -> VBitSet<StdVal<MIN, MAX>> {
    let mut result = VBitSet::<StdVal<MIN, MAX>>::empty();
    let mut wn = whisper_neighbors::<MIN, MAX>();
    for v in StdVal::<MIN, MAX>::possibilities() {
        let neighbors = wn.get(dist, v);
        let ln = left.intersection(&neighbors);
        let rn = right.intersection(&neighbors);
        if !ln.is_empty() && !rn.is_empty() {
            result.insert(&v);
        }
    }
    result
}

// TODO: Eventually add a generic implementation for whisper-type constraints.
// Currently just putting whisper-wide utility code here.

#[cfg(test)]
mod test {
    use crate::sudoku::unpack_stdval_vals;
    use super::*;

    fn assert_neighbors<const MIN: u8, const MAX: u8, const DIST: u8>(
        val: u8, neighbors: Vec<u8>,
    ) {
        let sval = StdVal::<MIN, MAX>::new(val);
        assert_eq!(
            unpack_stdval_vals::<MIN, MAX, _>(&whisper_neighbors().get(DIST, sval)),
            neighbors,
            "Neighbors for {} with distance {} should be {:?}",
            val, DIST, neighbors
        );
    }

    #[test]
    fn test_whisper_neighbors_gw() {
        assert_neighbors::<1, 9, 5>(1, vec![6, 7, 8, 9]);
        assert_neighbors::<1, 9, 5>(2, vec![7, 8, 9]);
        assert_neighbors::<1, 9, 5>(3, vec![8, 9]);
        assert_neighbors::<1, 9, 5>(4, vec![9]);
        assert_neighbors::<1, 9, 5>(5, vec![]);
        assert_neighbors::<1, 9, 5>(6, vec![1]);
        assert_neighbors::<1, 9, 5>(7, vec![1, 2]);
        assert_neighbors::<1, 9, 5>(8, vec![1, 2, 3]);
        assert_neighbors::<1, 9, 5>(9, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_whisper_neighbors_dw() {
        assert_neighbors::<1, 9, 4>(1, vec![5, 6, 7, 8, 9]);
        assert_neighbors::<1, 9, 4>(2, vec![6, 7, 8, 9]);
        assert_neighbors::<1, 9, 4>(3, vec![7, 8, 9]);
        assert_neighbors::<1, 9, 4>(4, vec![8, 9]);
        assert_neighbors::<1, 9, 4>(5, vec![1, 9]);
        assert_neighbors::<1, 9, 4>(6, vec![1, 2]);
        assert_neighbors::<1, 9, 4>(7, vec![1, 2, 3]);
        assert_neighbors::<1, 9, 4>(8, vec![1, 2, 3, 4]);
        assert_neighbors::<1, 9, 4>(9, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_whisper_neighbors_other() {
        // This isn't a common rule-set but it should still work.
        assert_neighbors::<1, 6, 3>(6, vec![1, 2, 3]);
        assert_neighbors::<1, 6, 3>(2, vec![5, 6]);
    }

    fn assert_possible_vals<const MIN: u8, const MAX: u8, const DIST: u8>(
        has_two_mutually_visible_neighbors: bool, vals: Vec<u8>,
    ) {
        assert_eq!(
            unpack_stdval_vals::<MIN, MAX, _>(&whisper_possible_values::<MIN, MAX>().get(DIST, has_two_mutually_visible_neighbors)),
            vals,
            "Possible vals for distance {} (has_two_mutually_visible_neighbors={}) should be {:?}",
            DIST, has_two_mutually_visible_neighbors, vals
        );
    }

    #[test]
    fn test_whisper_possible_vals_gw() {
        assert_possible_vals::<1, 9, 5>(
            false,
            vec![1, 2, 3, 4, 6, 7, 8, 9],
        );
        assert_possible_vals::<1, 9, 5>(
            true,
            vec![1, 2, 3, 7, 8, 9],
        );
    }

    #[test]
    fn test_whisper_possible_vals_dw() {
        assert_possible_vals::<1, 9, 4>(
            false,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
        );
        assert_possible_vals::<1, 9, 4>(
            true,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
        );
    }

    fn assert_between<const MIN: u8, const MAX: u8, const DIST: u8>(
        left: Vec<u8>, right: Vec<u8>, expected: Vec<u8>,
    ) {
        let mut left_set = VBitSet::<StdVal<MIN, MAX>>::empty();
        for v in &left {
            left_set.insert(&StdVal::<MIN, MAX>::new(*v));
        }
        let mut right_set = VBitSet::<StdVal<MIN, MAX>>::empty();
        for v in &right {
            right_set.insert(&StdVal::<MIN, MAX>::new(*v));
        }
        let result = whisper_between::<MIN, MAX, _>(DIST, &left_set, &right_set);
        assert_eq!(
            unpack_stdval_vals::<MIN, MAX, _>(&result),
            expected,
            "Values that fit between {:?} and {:?} with distance {} should be {:?}",
            left, right, DIST, expected
        )
    }
    
    #[test]
    fn test_whisper_between_gw() {
        assert_between::<1, 9, 5>(vec![1], vec![2], vec![7, 8, 9]);
        assert_between::<1, 9, 5>(vec![3], vec![4], vec![9]);
        assert_between::<1, 9, 5>(vec![1], vec![9], vec![]);
        assert_between::<1, 9, 5>(vec![6], vec![7], vec![1]);
        assert_between::<1, 9, 5>(vec![8], vec![9], vec![1, 2, 3]);
        assert_between::<1, 9, 5>(vec![1, 2, 3], vec![3, 4], vec![8, 9]);
        assert_between::<1, 9, 5>(vec![7, 8], vec![8, 9], vec![1, 2, 3]);
    }

    #[test]
    fn test_whisper_between_dw() {
        assert_between::<1, 9, 4>(vec![1], vec![2], vec![6, 7, 8, 9]);
        assert_between::<1, 9, 4>(vec![3], vec![4], vec![8, 9]);
        assert_between::<1, 9, 4>(vec![1], vec![9], vec![5]);
        assert_between::<1, 9, 4>(vec![6], vec![7], vec![1, 2]);
        assert_between::<1, 9, 4>(vec![8], vec![9], vec![1, 2, 3, 4]);
        assert_between::<1, 9, 4>(vec![1, 2, 3], vec![3, 4], vec![7, 8, 9]);
        assert_between::<1, 9, 4>(vec![7, 8], vec![8, 9], vec![1, 2, 3, 4]);
    }
}