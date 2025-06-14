use std::{collections::HashMap, sync::{LazyLock, Mutex}};
use crate::{core::{empty_set, filled_map, UVMap, UVSet, Value}, sudoku::SVal};

/// Tables of useful sets for whisper-style constraints.
static WHISPER_NEIGHBORS: LazyLock<Mutex<HashMap<(u8, u8, u8), UVMap<u8, UVSet<u8>>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});
static WHISPER_POSSIBLE_VALS: LazyLock<Mutex<HashMap<(u8, u8, u8, bool), UVSet<u8>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

pub fn whisper_neighbors<const MIN: u8, const MAX: u8>(dist: u8, val: SVal<MIN, MAX>) -> UVSet<u8> {
    let mut map = WHISPER_NEIGHBORS.lock().unwrap();
    let inner_map = map.entry((MIN, MAX, dist)).or_insert_with(|| {
        let mut neighbors = filled_map::<u8, SVal<MIN, MAX>, UVSet<u8>>(empty_set::<u8, SVal<MIN, MAX>>());
        for v1 in SVal::<MIN, MAX>::possibilities() {
            for v2 in SVal::<MIN, MAX>::possibilities() {
                if v1 == v2 {
                    continue;
                }
                let d = v1.val().abs_diff(v2.val());
                if d >= dist {
                    neighbors.get_mut(v1.to_uval()).insert(v2.to_uval());
                }
            }
        }
        neighbors
    });
    inner_map.get(val.to_uval()).clone()
}

pub fn whisper_possible_values<const MIN: u8, const MAX: u8>(dist: u8, has_two_mutually_visible_neighbors: bool) -> UVSet<u8> {
    let mut map = WHISPER_POSSIBLE_VALS.lock().unwrap();
    map.entry((MIN, MAX, dist, has_two_mutually_visible_neighbors)).or_insert_with(|| {
        let mut possible_vals = empty_set::<u8, SVal<MIN, MAX>>();
        for v in SVal::<MIN, MAX>::possibilities() {
            let neighbors = whisper_neighbors::<MIN, MAX>(dist, v);
            if neighbors.len() >= 2 || (!neighbors.is_empty() && !has_two_mutually_visible_neighbors) {
                possible_vals.insert(v.to_uval());
            }
        }
        possible_vals
    }).clone()
}

pub fn whisper_between<const MIN: u8, const MAX: u8>(dist: u8, left: &UVSet<u8>, right: &UVSet<u8>) -> UVSet<u8> {
    let mut result = empty_set::<u8, SVal<MIN, MAX>>();
    for v in SVal::<MIN, MAX>::possibilities() {
        let mut ln = whisper_neighbors(dist, v);
        let mut rn = ln.clone();
        ln.intersect_with(left);
        rn.intersect_with(right);
        if !ln.is_empty() && !rn.is_empty() {
            result.insert(v.to_uval());
        }
    }
    result
}

// TODO: Eventually add a generic implementation for whisper-type constraints.
// Currently just putting whisper-wide utility code here.

#[cfg(test)]
mod test {
    use crate::sudoku::unpack_sval_vals;
    use super::*;

    fn assert_neighbors<const MIN: u8, const MAX: u8, const DIST: u8>(
        val: u8, neighbors: Vec<u8>,
    ) {
        let sval = SVal::<MIN, MAX>::new(val);
        assert_eq!(
            unpack_sval_vals::<MIN, MAX>(&whisper_neighbors(DIST, sval)),
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
            unpack_sval_vals::<MIN, MAX>(&whisper_possible_values::<MIN, MAX>(DIST, has_two_mutually_visible_neighbors)),
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
        let mut left_set = empty_set::<u8, SVal<MIN, MAX>>();
        for v in &left {
            left_set.insert(SVal::<MIN, MAX>::new(*v).to_uval());
        }
        let mut right_set = empty_set::<u8, SVal<MIN, MAX>>();
        for v in &right {
            right_set.insert(SVal::<MIN, MAX>::new(*v).to_uval());
        }
        let result = whisper_between::<MIN, MAX>(DIST, &left_set, &right_set);
        assert_eq!(
            unpack_sval_vals::<MIN, MAX>(&result),
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