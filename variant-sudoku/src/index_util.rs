use std::{collections::hash_map::HashMap, hash::Hash};
use crate::core::{Error, Index, VGrid, Value};

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

pub fn check_orthogonally_adjacent(c1: Index, c2: Index) -> Result<(), String> {
    let diff = (c1[0].abs_diff(c2[0]), c1[1].abs_diff(c2[1]));
    if diff != (0, 1) && diff != (1, 0) {
        Err(format!("Cells {:?} and {:?} are not orthogonally adjacent", c1, c2))
    } else {
        Ok(())
    }
}

pub fn check_adjacent(c1: Index, c2: Index) -> Result<(), String> {
    let diff = (c1[0].abs_diff(c2[0]), c1[1].abs_diff(c2[1]));
    if diff != (0, 1) && diff != (1, 0) && diff != (1, 1) {
        Err(format!("Cells {:?} and {:?} are not adjacent", c1, c2))
    } else {
        Ok(())
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

fn norm_delta(from_u: usize, to_u: usize) -> i32 {
    if from_u < to_u {
        1
    } else if from_u > to_u {
        -1
    } else {
        0
    }
}

fn get_dir(from_cell: Index, to_cell: Index) -> Result<[i32; 2], String> {
    let dir = [norm_delta(from_cell[0], to_cell[0]), norm_delta(from_cell[1], to_cell[1])];
    if dir[0] == 0 || dir[1] == 0 {
        Ok(dir)
    } else if from_cell[0].abs_diff(to_cell[0]) == from_cell[1].abs_diff(to_cell[1]) {
        Ok(dir)
    } else {
        Err(format!("Direction from {:?} to {:?} is neither cardinal nor perfectly diagonal", from_cell, to_cell))
    }
}

fn get_orth_dir(from_cell: Index, to_cell: Index) -> Result<[i32; 2], String> {
    let dir = [norm_delta(from_cell[0], to_cell[0]), norm_delta(from_cell[1], to_cell[1])];
    if dir[0] == 0 || dir[1] == 0 {
        Ok(dir)
    } else {
        Err(format!("Direction from {:?} to {:?} is not a cardinal direction", from_cell, to_cell))
    }
}

// Notes:
// - Half-open (leaves out the end point)
// - Precondition that repeatedly adding dir to start will eventually
//   reach end.
fn cell_range(start: Index, end: Index, dir: [i32; 2]) -> Vec<Index> {
    let mut cells = vec![];
    let mut cur = start;
    while cur != end {
        cells.push(cur);
        cur[0] = (cur[0] as i32 + dir[0]) as usize;
        cur[1] = (cur[1] as i32 + dir[1]) as usize;
    }
    cells
}

pub fn expand_orthogonal_polyline(vertices: Vec<Index>) -> Result<Vec<Index>, String> {
    let mut cells = vec![];
    if vertices.len() < 2 {
        return Err(format!("A polyline must have at least 2 vertices, but got: {:?}", vertices))
    }
    for i in 0..(vertices.len()-1) {
        let dir = get_orth_dir(vertices[i], vertices[i+1])?;
        cells.extend(cell_range(vertices[i], vertices[i+1], dir));
    }
    cells.push(*vertices.last().unwrap());
    Ok(cells)
}

pub fn expand_polyline(vertices: Vec<Index>) -> Result<Vec<Index>, String> {
    let mut cells = vec![];
    if vertices.len() < 2 {
        return Err(format!("A polyline must have at least 2 vertices, but got: {:?}", vertices))
    }
    for i in 0..(vertices.len()-1) {
        let dir = get_dir(vertices[i], vertices[i+1])?;
        cells.extend(cell_range(vertices[i], vertices[i+1], dir));
    }
    cells.push(*vertices.last().unwrap());
    Ok(cells)
}

fn is_box_char(c: char) -> bool {
    if c == '-' || c == '|' || c == '+' {
        // Pure ascii box-drawing characters
        true
    } else {
        // Unicode ones
        '\u{2500}' <= c && c <= '\u{257f}'
    }
}

fn split_and_filter_grid_line(s: &str) -> Vec<String> {
    let by_ws_and_box: Vec<String> = s
        .split(|c| is_box_char(c) || c.is_whitespace())
        .filter(|word| !word.is_empty())
        .map(|word| word.to_string())
        .collect();
    // The dense case
    if by_ws_and_box.len() == 1 {
        return s.chars()
            .map(|c| c.to_string())
            .collect();
    }
    by_ws_and_box
}

pub fn parse_grid(s: &str, rows: usize, cols: usize) -> Result<Vec<Vec<String>>, Error> {
    let mut grid: Vec<Vec<String>> = vec![vec!["".to_string(); cols]; rows];
    let lines: Vec<Vec<String>> = s.lines()
        .map(split_and_filter_grid_line)
        .filter(|w| !w.is_empty())
        .collect();
    if lines.len() != rows {
        return Err(Error::new(format!(
            "Not enough (non-trivial) rows: {} (expected at least {})", lines.len(), rows,
        )));
    }
    for (r, line) in lines.iter().enumerate() {
        if line.len() != cols {
            return Err(Error::new(format!(
                "Invalid number of cols: {} (expected {})", line.len(), cols,
            )));
        }
        for (c, s) in line.iter().enumerate() {
            grid[r][c] = s.clone();
        }
    }
    Ok(grid)
}

pub fn parse_val_grid<V: Value>(s: &str, rows: usize, cols: usize) -> Result<VGrid<V>, Error> {
    let parsed = parse_grid(s, rows, cols)?;
    let mut grid = VGrid::<V>::new(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            // "." represents None, which already is present in the grid.
            if parsed[r][c] != "." {
                let v = V::parse(parsed[r][c].as_str())?;
                grid.set([r, c], Some(v));
            }
        }
    }
    Ok(grid)
}

pub fn parse_region_grid(s: &str, rows: usize, cols: usize) -> Result<HashMap<String, Vec<Index>>, Error> {
    let grid = parse_grid(s, rows, cols)?;
    let mut sym_to_indices: HashMap<String, Vec<Index>> = HashMap::new();
    for r in 0..rows {
        for c in 0..cols {
            sym_to_indices.entry(grid[r][c].clone()).or_default().push([r, c]);
        }
    }
    Ok(sym_to_indices)
}

#[cfg(test)]
mod test {
    use super::*;

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
    fn test_check_orthogonally_adacent() {
        // 4 cardinal directions = yes
        assert!(check_orthogonally_adjacent([2, 2], [1, 2]).is_ok());
        assert!(check_orthogonally_adjacent([2, 2], [3, 2]).is_ok());
        assert!(check_orthogonally_adjacent([2, 2], [2, 1]).is_ok());
        assert!(check_orthogonally_adjacent([2, 2], [2, 3]).is_ok());
        // 4 diagonal directions = no
        assert!(check_orthogonally_adjacent([2, 2], [1, 1]).is_err());
        assert!(check_orthogonally_adjacent([2, 2], [1, 3]).is_err());
        assert!(check_orthogonally_adjacent([2, 2], [3, 1]).is_err());
        assert!(check_orthogonally_adjacent([2, 2], [3, 3]).is_err());
        // Longer jump in any direction = no
        assert!(check_orthogonally_adjacent([2, 2], [0, 0]).is_err());
        assert!(check_orthogonally_adjacent([2, 2], [0, 4]).is_err());
        assert!(check_orthogonally_adjacent([2, 2], [4, 2]).is_err());
        assert!(check_orthogonally_adjacent([2, 2], [2, 0]).is_err());
    }

    #[test]
    fn test_check_adacent() {
        // 4 cardinal directions = yes
        assert!(check_adjacent([2, 2], [1, 2]).is_ok());
        assert!(check_adjacent([2, 2], [3, 2]).is_ok());
        assert!(check_adjacent([2, 2], [2, 1]).is_ok());
        assert!(check_adjacent([2, 2], [2, 3]).is_ok());
        // 4 diagonal directions = yes
        assert!(check_adjacent([2, 2], [1, 1]).is_ok());
        assert!(check_adjacent([2, 2], [1, 3]).is_ok());
        assert!(check_adjacent([2, 2], [3, 1]).is_ok());
        assert!(check_adjacent([2, 2], [3, 3]).is_ok());
        // Longer jump in any direction = no
        assert!(check_adjacent([2, 2], [0, 0]).is_err());
        assert!(check_adjacent([2, 2], [0, 4]).is_err());
        assert!(check_adjacent([2, 2], [4, 2]).is_err());
        assert!(check_adjacent([2, 2], [2, 0]).is_err());
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

    #[test]
    fn test_expand_orthogonal_polyline() {
        assert_eq!(
            expand_orthogonal_polyline(vec![[2, 2], [0, 2]]).unwrap(),
            vec![[2, 2], [1, 2], [0, 2]],
        );
        assert_eq!(
            expand_orthogonal_polyline(vec![[2, 2], [4, 2]]).unwrap(),
            vec![[2, 2], [3, 2], [4, 2]],
        );
        assert_eq!(
            expand_orthogonal_polyline(vec![[2, 2], [2, 0]]).unwrap(),
            vec![[2, 2], [2, 1], [2, 0]],
        );
        assert_eq!(
            expand_orthogonal_polyline(vec![[2, 2], [2, 4]]).unwrap(),
            vec![[2, 2], [2, 3], [2, 4]],
        );
        // Exact and messed up diagonals are both bad
        assert!(expand_orthogonal_polyline(vec![[2, 2], [4, 4]]).is_err());
        assert!(expand_orthogonal_polyline(vec![[2, 2], [4, 5]]).is_err());
    }

    #[test]
    fn test_expand_polyline() {
        assert_eq!(
            expand_polyline(vec![[2, 2], [0, 2]]).unwrap(),
            vec![[2, 2], [1, 2], [0, 2]],
        );
        assert_eq!(
            expand_polyline(vec![[2, 2], [4, 2]]).unwrap(),
            vec![[2, 2], [3, 2], [4, 2]],
        );
        assert_eq!(
            expand_polyline(vec![[2, 2], [2, 0]]).unwrap(),
            vec![[2, 2], [2, 1], [2, 0]],
        );
        assert_eq!(
            expand_polyline(vec![[2, 2], [2, 4]]).unwrap(),
            vec![[2, 2], [2, 3], [2, 4]],
        );
        // Exact diagonals are ok
        assert_eq!(
            expand_polyline(vec![[2, 2], [0, 0]]).unwrap(),
            vec![[2, 2], [1, 1], [0, 0]],
        );
        assert_eq!(
            expand_polyline(vec![[2, 2], [4, 0]]).unwrap(),
            vec![[2, 2], [3, 1], [4, 0]],
        );
        assert_eq!(
            expand_polyline(vec![[2, 2], [0, 4]]).unwrap(),
            vec![[2, 2], [1, 3], [0, 4]],
        );
        assert_eq!(
            expand_polyline(vec![[2, 2], [4, 4]]).unwrap(),
            vec![[2, 2], [3, 3], [4, 4]],
        );
        // Messed up diagonals are still bad
        assert!(expand_polyline(vec![[2, 2], [4, 5]]).is_err());
    }

    #[test]
    fn test_parse_grid_dense() {
        let input: &str = "abcd\n\
                           .123\n\
                           ..vx\n\
                           ...$\n";
        let parsed = parse_grid(input, 4, 4).unwrap();
        assert_eq!(parsed[0], vec!["a", "b", "c", "d"]);
        assert_eq!(parsed[3], vec![".", ".", ".", "$"]);
    }

    #[test]
    fn test_parse_grid_pretty_based_on_spaces() {
        let input: &str = "a b c d\n\
                           . 1 2 3\n\
                           . . v x\n\
                           . . . $\n";
        let parsed = parse_grid(input, 4, 4).unwrap();
        assert_eq!(parsed[0], vec!["a", "b", "c", "d"]);
        assert_eq!(parsed[3], vec![".", ".", ".", "$"]);
    }

    #[test]
    fn test_parse_grid_pretty_ignores_grid_lines() {
        let input: &str = "+------+------+\n\
                           | a  b | c  d |\n\
                           | .  1 | 2  3 |\n\
                           |------+------|\n\
                           | .  . | v  x |\n\
                           | .  . | .  $ |\n\
                           +------+------|\n";
        let parsed = parse_grid(input, 4, 4).unwrap();
        assert_eq!(parsed[0], vec!["a", "b", "c", "d"]);
        assert_eq!(parsed[3], vec![".", ".", ".", "$"]);
    }
}