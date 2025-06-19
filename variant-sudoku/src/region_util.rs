
use std::{collections::hash_map::HashMap, hash::Hash};
use crate::core::Index;

struct DisjointSet<T: Eq + Clone + Hash> {
    parents: HashMap<T, T>,
}

impl <T: Eq + Clone + Hash> DisjointSet<T> {
    pub fn new() -> Self {
        Self { parents: HashMap::new() }
    }

    pub fn insert(&mut self, t: &T) {
        self.parents.insert(t.clone(), t.clone());
    }

    pub fn contains(&self, t: &T) -> bool {
        self.parents.contains_key(t)
    }

    pub fn unite(&mut self, t1: &T, t2: &T) {
        assert!(self.parents.contains_key(t1));
        assert!(self.parents.contains_key(t2));
        let p1 = self.find(t1);
        let p2 = self.find(t2);
        self.parents.get_mut(&p1).map(|p| { *p = p2 });
    }

    pub fn find(&mut self, t: &T) -> T {
        assert!(self.parents.contains_key(t));
        let parent = self.parents.get(t).unwrap().clone();
        if *t == parent {
            parent
        } else {
            let p = self.find(&parent);
            self.parents.get_mut(t).map(|tp| *tp = p.clone());
            p
        }
    }
}

pub fn check_orthogonally_connected(cells: &Vec<Index>) -> Result<(), String> {
    let mut uf: DisjointSet<Index> = DisjointSet::new();
    for cell in cells {
        let [r, c] = *cell;
        uf.insert(&[r, c]);
        if r > 0 && uf.contains(&[r-1, c]) {
            uf.unite(&[r-1, c], &[r, c]);
        }
        if uf.contains(&[r+1, c]) {
            uf.unite(&[r+1, c], &[r, c]);
        }
        if c > 0 && uf.contains(&[r, c-1]) {
            uf.unite(&[r, c-1], &[r, c]);
        }
        if uf.contains(&[r, c+1]) {
            uf.unite(&[r, c+1], &[r, c]);
        }
    }
    let mut rep = None;
    for cell in cells {
        let p = uf.find(cell);
        if let Some(r) = rep {
            if r != p {
                return Err(format!(
                    "Cell {:?} belongs to a different orthogonally connected \
                     component than {:?}", *cell, r,
                ));
            }
        } else {
            rep = Some(p)
        }
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::region_util::{check_orthogonally_connected, DisjointSet};

    #[test]
    fn test_uf() {
        let mut uf = DisjointSet::new();
        uf.insert(&[0, 0]);
        assert!(uf.contains(&[0, 0]));
        assert_eq!(uf.find(&[0, 0]), [0, 0]);
        uf.insert(&[0, 1]);
        assert!(uf.contains(&[0, 1]));
        assert_eq!(uf.find(&[0, 1]), [0, 1]);
        uf.insert(&[0, 2]);
        assert!(uf.contains(&[0, 2]));
        assert_eq!(uf.find(&[0, 2]), [0, 2]);
        uf.unite(&[0, 0], &[0, 2]);
        uf.unite(&[0, 2], &[0, 1]);
        assert_eq!(uf.find(&[0, 0]), uf.find(&[0, 1]));
        assert_eq!(uf.find(&[0, 1]), uf.find(&[0, 2]));
    }

    #[test]
    fn test_check_orthogonally_connected() {
        assert!(check_orthogonally_connected(&vec![
            [0, 0], [0, 1], [0, 3], [0, 4],
        ]).is_err());
        assert!(check_orthogonally_connected(&vec![
            [0, 0], [0, 1], [0, 3], [0, 4], [1, 1], [1, 3], [1, 2],
        ]).is_ok());
    }
}