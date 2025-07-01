
pub fn color_ave2(a: (u8, u8, u8), b: (u8, u8, u8)) -> (u8, u8, u8) {
    let r = ((a.0 as f32)*0.5 + (b.0 as f32)*0.5) as u8;
    let g = ((a.1 as f32)*0.5 + (b.1 as f32)*0.5) as u8;
    let b = ((a.2 as f32)*0.5 + (b.2 as f32)*0.5) as u8;
    (r, g, b)
}

pub fn color_ave(colors: &Vec<(u8, u8, u8)>) -> (u8, u8, u8) {
    let sum = colors.iter().fold((0.0, 0.0, 0.0), |a, b| {
        (a.0 + b.0 as f32, a.1 + b.1 as f32, a.2 + b.2 as f32)
    });
    let n = colors.len() as f32;
    ((sum.0/n) as u8, (sum.1/n) as u8, (sum.2/n) as u8)
}

pub fn color_scale(c: (u8, u8, u8), ratio: f32) -> (u8, u8, u8) {
    color_lerp((0, 0, 0), c, ratio)
}

pub fn color_lerp(a: (u8, u8, u8), b: (u8, u8, u8), ratio: f32) -> (u8, u8, u8) {
    let (t, tc) = (ratio, 1.0 - ratio);
    let r = ((a.0 as f32)*tc + (b.0 as f32)*t) as u8;
    let g = ((a.1 as f32)*tc + (b.1 as f32)*t) as u8;
    let b = ((a.2 as f32)*tc + (b.2 as f32)*t) as u8;
    (r, g, b)
}

// Green is low, orange is high, pink is exact middle, if any.
pub fn color_polarity(min: u8, max: u8, v: u8) -> (u8, u8, u8) {
    let mid = ((min as f32) + (max as f32))/2.0;
    if (v as f32) < mid {
        (0, 200, 0)
    } else if (v as f32) > mid {
        (200, 130, 0)
    } else {
        (200, 0, 200)
    }
}

pub fn color_dist(a: (u8, u8, u8), b: (u8, u8, u8)) -> f32 {
    let dr = a.0 as i16 - b.0 as i16;
    let dg = a.1 as i16 - b.1 as i16;
    let db = a.2 as i16 - b.2 as i16;
    ((dr * dr + dg * dg + db * db) as f32).sqrt()
}

pub fn color_clamp_i32(c: (i32, i32, i32)) -> (u8, u8, u8) {
    (
        c.0.clamp(0, 255) as u8,
        c.1.clamp(0, 255) as u8,
        c.2.clamp(0, 255) as u8,
    )
}

pub fn color_clamp_f32(c: (f32, f32, f32)) -> (u8, u8, u8) {
    (
        c.0.clamp(0.0, 255.0) as u8,
        c.1.clamp(0.0, 255.0) as u8,
        c.2.clamp(0.0, 255.0) as u8,
    )
}

// Generate a palette of multiple similar colors. Uses the fibonacci sphere
// algorithm to distribute them evenly-ish on the surface of the sphere with
// the given radius, centered at the prototype.
pub fn color_fib_palette(prototype: (u8, u8, u8), n: usize, radius: f32) -> Vec<(u8, u8, u8)> {
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut palette = Vec::with_capacity(n);
    for i in 0..n {
        let i = i as f32;
        let n = n as f32;
        let theta = 2.0 * std::f32::consts::PI * i / phi;
        let z = 1.0 - (2.0 * i + 1.0) / n;
        let r = (1.0 - z * z).sqrt();
        let x = r * theta.cos();
        let y = r * theta.sin();
        let dx = (x * radius).round() as i32;
        let dy = (y * radius).round() as i32;
        let dz = (z * radius).round() as i32;
        let rgb = color_clamp_i32((
            prototype.0 as i32 + dx,
            prototype.1 as i32 + dy,
            prototype.2 as i32 + dz,
        ));
        palette.push(rgb);
    }
    palette
}

struct ColorGraph {
    n: usize,
    edges: Vec<Vec<usize>>,
    enabled: Vec<bool>,
    color_ids: Vec<Option<u8>>,
}

impl ColorGraph {
    fn new(edges: Vec<Vec<usize>>) -> Self {
        let n = edges.len();
        let enabled = vec![true; n];
        let color_ids = vec![None; n];
        let max_id = edges.iter()
            .map(|ids| {
                ids.iter().fold(0 as usize, |i, j| i.max(*j))
            })
            .fold(0 as usize, |i, j| { i.max(j) });
        if max_id >= n {
            panic!("Invalid node id found in edges: {} (max acceptable is {}", max_id, n-1);
        }
        Self {
            n,
            edges,
            enabled,
            color_ids,
        }
    }

    fn degree(&self, i: usize) -> usize {
        if !self.enabled[i] {
            panic!("Cannot call degree({}) on disabled node", i);
        }
        self.edges[i].iter()
            .filter(|id| self.enabled[**id])
            .count()
    }

    fn is_empty(&self) -> bool {
        (0..self.n).all(|i| !self.enabled[i])
    }

    fn find_max5_node(&self) -> usize {
        if self.is_empty() {
            panic!("Cannot call find_max5_node on an empty graph");
        }
        for i in 0..self.n {
            if self.enabled[i] && self.degree(i) <= 5 {
                return i;
            }
        }
        panic!("Planarity violated: no node with degree <= 5 found!");
    }

    fn color(&mut self) {
        self.color_ids = vec![None; self.n];
        let mut stack = Vec::with_capacity(self.n);
        while !self.is_empty() {
            let id = self.find_max5_node();
            stack.push(id);
            self.enabled[id] = false;
        }
        while !stack.is_empty() {
            let id = stack.pop().unwrap();
            let neighbor_colors = self.edges[id].iter().filter_map(|neighbor| {
                self.color_ids[*neighbor]
            }).collect::<Vec<_>>();
            for color_id in 0..6 {
                if !neighbor_colors.contains(&color_id) {
                    self.color_ids[id] = Some(color_id);
                    break;
                }
            }
            if self.color_ids[id].is_none() {
                panic!("Unexpectedly failed to color node: {}", id);
            }
            self.enabled[id] = true;
        }
    }
}

/// Helper that assists in calling color_planar_graph.
pub fn find_undirected_edges<T, F: Fn(&T, &T) -> bool>(nodes: &Vec<T>, adjacent: F) -> Vec<Vec<usize>> {
    let mut edges = vec![Vec::new(); nodes.len()];
    for i in 0..(nodes.len()-1) {
        for j in (i+1)..nodes.len() {
            if adjacent(&nodes[i], &nodes[j]) {
                edges[i].push(j);
                edges[j].push(i);
            }
        }
    }
    for i in 0..nodes.len() {
        edges[i].sort();
    }
    edges
}

/// Color a planar graph using a max of five colors.
/// Notes:
/// - N = edges.len() = the number of nodes
/// - edges must be a valid graph in adjacency list form for a graph of size
///   N with ids 0..N
/// - palette.len() >= 5
/// - return value is len = N
pub fn color_planar_graph(edges: Vec<Vec<usize>>, palette: &Vec<(u8, u8, u8)>) -> Vec<(u8, u8, u8)> {
    let mut g = ColorGraph::new(edges);
    if palette.len() < 5 {
        panic!("color_planar_graph requires palettes with at least 5 colors; got {}", palette.len());
    }
    g.color();
    g.color_ids.iter().map(|color_id| {
        if let Some(cid) = color_id {
            palette[*cid as usize]
        } else {
            panic!("Unexpected uncolored node!")
        }
    }).collect()
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use super::*;

    #[test]
    fn test_color_planar_graph() {
        //    D
        //  / | \
        // A--C--E--F
        //  \ | /
        //    B
        let readable_graph = HashMap::from([
            ('A', vec!['B', 'C', 'D']),
            ('B', vec!['A', 'C', 'E']),
            ('C', vec!['A', 'B', 'D', 'E']),
            ('D', vec!['A', 'C', 'E']),
            ('E', vec!['B', 'C', 'D', 'F']),
            ('F', vec!['E']),
        ]);
        let rgbbw = vec![
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255),
        ];
        let colors = color_planar_graph(
            find_undirected_edges(
                &vec!['A', 'B', 'C', 'D', 'E', 'F'],
                |a, b| {
                    readable_graph.get(a).unwrap().contains(b)
                },
            ),
            &rgbbw,
        );
        let color_map = ['A', 'B', 'C', 'D', 'E', 'F']
            .iter().zip(colors)
            .map(|(id, c)| (*id, c))
            .collect::<HashMap<_, _>>();
        for id in ['A', 'B', 'C', 'D', 'E', 'F'] {
            let c = color_map[&id];
            assert!(rgbbw.contains(&c));
            for neighbor in &readable_graph[&id] {
                let c2 = color_map[neighbor];
                assert_ne!(c, c2);
            }
        }
    }
}
