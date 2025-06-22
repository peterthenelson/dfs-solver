use std::collections::HashMap;
use std::sync::Mutex;
use std::{borrow::Cow, marker::PhantomData};
use std::fmt::{Debug, Display};
use bit_set::BitSet;
use num::{PrimInt, Unsigned};

/// Error type. This is used to indicate something wrong with either the
/// puzzle/strategy/constraints or with the algorithm itself. Violations of
/// constraints or exhaustion of the search space are not errors.
#[derive(Debug, Clone, PartialEq)]
pub struct Error(Cow<'static, str>);
impl Error {
    pub const fn new_const(s: &'static str) -> Self {
        Error(Cow::Borrowed(s))
    }

    pub fn new<S: Into<String>>(s: S) -> Self {
        Error(Cow::Owned(s.into()))
    }
}

/// Puzzles are made up of a grid of cells, each of which has some value drawn
/// from a finite set of possible values. (Non-rectangular or multi-layer grids
/// can be simulated by the puzzle implementation, but the underlying structure
/// is always a rectangular grid of cells.)
pub type Index = [usize; 2];

pub trait GridIndex {
    // Is the index still valid or has it gone off the end of the grid?
    fn in_bounds(&self, rows: usize, cols: usize) -> bool;
    // Increment the index (supposing a grid of given dimensions).
    fn increment(&mut self, rows: usize, cols: usize);
}

impl GridIndex for Index {
    fn in_bounds(&self, rows: usize, cols: usize) -> bool {
        self[0] < rows && self[1] < cols
    }

    fn increment(&mut self, rows: usize, cols: usize) {
        let _ = rows;
        self[1] += 1;
        if self[1] >= cols {
            self[1] = 0;
            self[0] += 1;
        }
    }
}

pub trait UInt: PrimInt + Unsigned + TryInto<usize> + Debug {
    fn from_usize(u: usize) -> Self;
    fn as_usize(&self) -> usize;
}
impl UInt for u8 {
    fn from_usize(u: usize) -> Self { u.try_into().unwrap() }
    fn as_usize(&self) -> usize { *self as usize }
}
impl UInt for u16 {
    fn from_usize(u: usize) -> Self { u.try_into().unwrap() }
    fn as_usize(&self) -> usize { *self as usize }
}
impl UInt for u32 { 
    fn from_usize(u: usize) -> Self { u.try_into().unwrap() }
    fn as_usize(&self) -> usize { *self as usize }
}
impl UInt for u64 { 
    fn from_usize(u: usize) -> Self { u.try_into().unwrap() }
    fn as_usize(&self) -> usize { *self as usize }
}
impl UInt for u128 { 
    fn from_usize(u: usize) -> Self { u.try_into().unwrap() }
    fn as_usize(&self) -> usize { *self as usize }
}

// Values in puzzles are implementation dependent, but they must be convertible
// to and from an unsigned integer type that ranges over some known (and small)
// cardinality. Instead of directly exposing UInts, we use a wrapper to avoid
// accidental misuse: These aren't the values you're looking for! They are just
// for containers that need to store them!
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UVWrapped;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UVUnwrapped;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UVal<U: UInt, S> {
    u: U,
    _state: PhantomData<S>,
}

impl <U: UInt> UVal<U, UVWrapped> {
    pub fn new(v: U) -> Self {
        UVal { u: v, _state: PhantomData }
    }

    pub(self) fn unwrap(self) -> UVal<U, UVUnwrapped> {
        UVal { u: self.u, _state: PhantomData }
    }
}

impl <U: UInt> UVal<U, UVUnwrapped> {
    pub fn value(&self) -> U {
        self.u
    }
}

/// Values in puzzles are drawn from a finite set of possible values. They are
/// represented as unsigned integers, but it's entirely up to the Value, State,
/// and Constraint implementations to interpret them.
pub trait Value: Copy + Clone + Display + Debug + PartialEq + Eq {
    type U: UInt;

    fn cardinality() -> usize;
    fn possibilities() -> Vec<Self>;
    fn nth(ord: usize) -> Self;
    fn parse(s: &str) -> Result<Self, Error>;

    fn ordinal(&self) -> usize;
    fn from_uval(u: UVal<Self::U, UVUnwrapped>) -> Self;
    fn to_uval(self) -> UVal<Self::U, UVWrapped>;
}

/// This is the underlying grid structure for a puzzle.
#[derive(Debug, Clone)]
pub struct UVGrid<U: UInt> {
    rows: usize,
    cols: usize,
    grid: Box<[Option<U>]>,
}

impl<U: UInt> UVGrid<U> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            grid: vec![None; rows * cols].into_boxed_slice(),
        }
    }

    pub fn get(&self, index: Index) -> Option<UVal<U, UVWrapped>> {
        self.grid[index[0] * self.cols + index[1]].map(|v| UVal::new(v))
    }

    pub fn set(&mut self, index: Index, value: Option<UVal<U, UVWrapped>>) {
        self.grid[index[0] * self.cols + index[1]] = value.map(|v| v.unwrap().value());
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// This is a mapping from values to some other type. It is densely represented
/// as a slice of values, indexed by their unsigned representations.
#[derive(Debug, Clone)]
pub struct UVMap<U: UInt, V: Clone> {
    vals: Box<[V]>,
    _marker: PhantomData<U>,
}

pub fn empty_map<K: Value, V: Clone + Default>() -> UVMap<K::U, V> {
    UVMap { vals: vec![V::default(); K::cardinality()].into_boxed_slice(), _marker: PhantomData }
}

pub fn filled_map<K: Value, V: Clone>(default: V) -> UVMap<K::U, V> {
    UVMap { vals: vec![default; K::cardinality()].into_boxed_slice(), _marker: PhantomData }
}

impl <U: UInt, V: Clone> UVMap<U, V> {
    pub fn get(&self, key: UVal<U, UVWrapped>) -> &V {
        &self.vals[key.unwrap().value().as_usize()]
    }

    pub fn get_mut(&mut self, key: UVal<U, UVWrapped>) -> &mut V {
        &mut self.vals[key.unwrap().value().as_usize()]
    }

    pub fn iter<K: Value<U = U>>(&self) -> std::vec::IntoIter<(K, V)> {
        self.vals.iter().enumerate().map(|(u, v)| {
            let k: K = to_value::<K>(UVal::<U, UVWrapped>::new(U::from_usize(u)));
            let v: V = v.clone();
            (k, v)
        }).collect::<Vec<(K, V)>>().into_iter()
    }
}

/// This a set of values (e.g., that are possible, that have been seen, etc.).
/// They are represented as a bitset of the possible values.
#[derive(Debug, Clone, PartialEq)]
pub struct UVSet<U: UInt> {
    s: BitSet,
    _marker: PhantomData<U>,
}

pub fn empty_set<V: Value>() -> UVSet<V::U> {
    UVSet {
        s: BitSet::with_capacity(V::cardinality()),
        _marker: PhantomData,
    }
}

fn leading_ones(n: usize) -> Vec<u8> {
    let full = n / 8;
    let remaining = n % 8;
    let mut result = vec![u8::MAX; full];
    if remaining > 0 {
        result.push(u8::MAX << (8 - remaining));
    }
    result
}

pub fn full_set<V: Value>() -> UVSet<V::U> {
    let n = V::cardinality();
    let mut s = UVSet {
        s: BitSet::with_capacity(n),
        _marker: PhantomData,
    };
    let ones = leading_ones(n);
    s.s.union_with(&BitSet::from_bytes(ones.as_slice()));
    s
}

pub fn pack_values<V: Value>(vals: &Vec<V>) -> UVSet<V::U> {
    let mut res = empty_set::<V>();
    for v in vals {
        res.insert(v.to_uval());
    }
    res
}

pub fn singleton_set<V: Value>(v: V) -> UVSet<V::U> {
    let mut s = empty_set::<V>();
    s.insert(v.to_uval());
    s
}

pub fn unpack_values<V: Value>(s: &UVSet<V::U>) -> Vec<V> {
    s.iter().map(|u| { to_value::<V>(u) }).collect::<Vec<_>>()
}

pub fn unpack_singleton<V: Value>(s: &UVSet<V::U>) -> Option<V> {
    if s.len() == 1 {
        Some(to_value::<V>(s.iter().next().unwrap()))
    } else {
        None
    }
}

pub fn unpack_first<V: Value>(s: &UVSet<V::U>) -> Option<V> {
    s.iter().next().map(|uv| to_value::<V>(uv))
}

pub fn unpack_last<V: Value>(s: &UVSet<V::U>) -> Option<V> {
    s.iter().last().map(|uv| to_value::<V>(uv))
}

impl <U: UInt> UVSet<U> {
    pub fn insert(&mut self, value: UVal<U, UVWrapped>) {
        self.s.insert(value.unwrap().value().as_usize());
    }

    pub fn remove(&mut self, value: UVal<U, UVWrapped>) {
        self.s.remove(value.unwrap().value().as_usize());
    }

    pub fn contains(&self, value: UVal<U, UVWrapped>) -> bool {
        self.s.contains(value.unwrap().value().as_usize())
    }

    pub fn is_empty(&self) -> bool {
        self.s.is_empty()
    }

    pub fn len(&self) -> usize {
        self.s.len()
    }

    pub fn clear(&mut self) {
        self.s.clear();
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = UVal<U, UVWrapped>> + 'a {
        self.s.iter().map(|i| UVal::new(U::from_usize(i)))
    }

    pub fn intersect_with(&mut self, other: &UVSet<U>) {
        self.s.intersect_with(&other.s);
    }

    pub fn intersection(&self, other: &UVSet<U>) -> UVSet<U> {
        let mut i = self.clone();
        i.s.intersect_with(&other.s);
        i
    }

    pub fn union(&self, other: &UVSet<U>) -> UVSet<U> {
        let mut u = self.clone();
        u.s.union_with(&other.s);
        u
    }

}

struct ConstStringRegistry {
    mapping: HashMap<&'static str, usize>,
    next_id: usize,
}

impl ConstStringRegistry {
    pub fn new() -> Self { Self { mapping: HashMap::new(), next_id: 0 } }
    pub fn register(&mut self, name: &'static str) -> usize {
        if let Some(id) = self.mapping.get(name) {
            *id
        } else {
            let id = self.next_id;
            self.mapping.insert(name, id);
            self.next_id += 1;
            id
        }
    }
    pub fn name(&self, id: usize) -> Option<&'static str> {
        for (name, feature_id) in self.mapping.iter() {
            if *feature_id == id {
                return Some(name);
            }
        }
        None
    }
}

/// Marker structs to indicate whether a compile-time string has already been
/// interned (or normalized to its usize representation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaybeId;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WithId;

lazy_static::lazy_static! {
    static ref FEATURE_REGISTRY: Mutex<ConstStringRegistry> = {
        Mutex::new(ConstStringRegistry::new())
    };
}

// NOTE: This is an expensive operation, so only use it for human-interface
// purposes (e.g., debugging, logging, etc.) and not during the solving process.
pub fn readable_key(id: usize) -> Option<Key<WithId>> {
    let registry = FEATURE_REGISTRY.lock().unwrap();
    registry.name(id).map(|name| {
        Key { name, id: Some(id), _state: PhantomData}
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key<S>{
    name: &'static str,
    id: Option<usize>,
    _state: PhantomData<S>,
}

impl <S> Key<S> {
    pub fn name(&self) -> &'static str { self.name }
}

impl Key<MaybeId> {
    // Keys are lazily initialized; the id is set when it is first used.
    pub fn new(name: &'static str) -> Self {
        Key { name, id: None, _state: PhantomData }
    }

    pub fn unwrap(&mut self) -> Key<WithId> {
        if let Some(id) = self.id {
            return Key { name: self.name, id: Some(id), _state: PhantomData };
        } else {
            let id = FEATURE_REGISTRY.lock().unwrap().register(self.name);
            self.id = Some(id);
            return Key { name: self.name, id: Some(id), _state: PhantomData };
        }
    }
}

impl Key<WithId> {
    pub fn id(&self) -> usize { self.id.unwrap() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FVMaybeNormed;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FVNormed;

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureVec<S> {
    features: Vec<(usize, f64)>,
    // Some methods only make sense when the features are sorted by id and
    // duplicate ids are dropped or merged.
    normalized: bool,
    _marker: PhantomData<S>,
}

impl FeatureVec<FVMaybeNormed> {
    pub fn new() -> Self {
        FeatureVec { features: Vec::new(), normalized: true, _marker: PhantomData }
    }

    pub fn from_pairs(weights: Vec<(&'static str, f64)>) -> Self {
        let mut fv = FeatureVec::new();
        for (k, v) in weights {
            fv.add(&Key::new(k).unwrap(), v);
        }
        fv
    }

    pub fn add(&mut self, key: &Key<WithId>, value: f64) {
        let id = key.id();
        self.features.push((id, value));
        self.normalized = false;
    }

    pub fn extend(&mut self, other: &FeatureVec<FVMaybeNormed>) {
        self.features.extend_from_slice(&other.features);
        self.normalized = false;
    }

    pub fn normalize(&mut self, combine: fn(usize, f64, f64) -> f64) {
        if self.normalized {
            return;
        } else if self.features.is_empty() {
            self.normalized = true;
            return;
        }
        self.features.sort_by(|a, b| a.0.cmp(&b.0));
        let mut dst = 0;
        let mut dst_id = self.features[dst].0;
        for src in 1..self.features.len() {
            let src_id = self.features[src].0;
            if src_id == dst_id {
                self.features[dst].1 = combine(src_id, self.features[dst].1, self.features[src].1);
            } else {
                dst += 1;
                self.features[dst] = self.features[src];
                dst_id = src_id;
            }
        }
        self.features.truncate(dst + 1);
        self.normalized = true;
    }

    pub fn try_normalized(&self) -> Result<&FeatureVec<FVNormed>, Error> {
        if self.normalized {
            Ok(unsafe { std::mem::transmute(self) })
        } else {
            Err(Error::new("FeatureVec is not normalized"))
        }
    }

    pub fn normalize_and(&mut self, combine: fn(usize, f64, f64) -> f64) -> &FeatureVec<FVNormed> {
        self.normalize(combine);
        unsafe { std::mem::transmute(self) }
    }
}

impl Default for FeatureVec<FVMaybeNormed> {
    fn default() -> Self {
        FeatureVec::new()
    }
}

impl Display for FeatureVec<FVMaybeNormed> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for (i, (k, v)) in self.features.iter().enumerate() {
            if let Some(feature) = readable_key(*k) {
                write!(f, "{} => {}", feature.name(), *v)?;
            } else {
                write!(f, "??? => {}", *v)?;
            }
            if i+1 < self.features.len() {
                write!(f, ", ")?;
            }
        }
        write!(f, "}}")
    }
}

impl FeatureVec<FVNormed> {
    pub fn get(&self, key: &Key<WithId>) -> Option<f64> {
        let id = key.id();
        for (k, v) in &self.features {
            if *k == id {
                return Some(*v);
            }
        }
        None
    }

    pub fn dot_product(&self, other: &FeatureVec<FVNormed>) -> f64 {
        let mut sum = 0.0;
        let mut i = 0;
        let mut j = 0;
        while i < self.features.len() && j < other.features.len() {
            if self.features[i].0 == other.features[j].0 {
                sum += self.features[i].1 * other.features[j].1;
                i += 1;
                j += 1;
            } else if self.features[i].0 < other.features[j].0 {
                i += 1;
            } else {
                j += 1;
            }
        }
        sum
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CertainDecision<V: Value> {
    pub index: Index,
    pub value: V,
}

impl <V: Value> CertainDecision<V> {
    pub fn new(index: Index, value: V) -> Self {
        Self { index, value }
    }
}

/// Constraints and ranking both may return early if they hit upon either a
/// contradiction or a certainty. This is a simple enum to represent this
/// short-circuiting.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintResult<V: Value> {
    Contradiction(Key<WithId>),
    Certainty(CertainDecision<V>, Key<WithId>),
    Ok,
}

/// When choosing to branch, we can either try all the possible values for a
/// particular cell, or we can try all possible cells for a particular value.
#[derive(Debug, Clone)]
pub enum BranchOver<V: Value> {
    Empty,
    Cell(Index, Vec<V>, usize),
    Value(V, Vec<Index>, usize),
}

/// A decision point in the puzzle. This includes the specific value that was
/// chosen, as well as the index of the cell that was modified, as well as the
/// alternative values/indices that have not been tried yet.
#[derive(Debug, Clone)]
pub struct BranchPoint<V: Value> {
    pub branch_step: usize,
    pub branch_attribution: Key<WithId>,
    pub choices: BranchOver<V>,
}

impl <V: Value> BranchPoint<V> {
    pub fn unique(step: usize, attribution: Key<WithId>, index: Index, value: V) -> Self {
        Self::for_cell(step, attribution, index, vec![value])
    }

    pub fn empty(step: usize, attribution: Key<WithId>) -> Self {
        BranchPoint { branch_step: step, branch_attribution: attribution, choices: BranchOver::Empty }
    }

    pub fn for_cell(step: usize, attribution: Key<WithId>, index: Index, values: Vec<V>) -> Self {
        if values.len() > 0 {
            BranchPoint {
                branch_step: step,
                branch_attribution: attribution,
                choices: BranchOver::Cell(index, values, 0),
            }
        } else {
            panic!("Cannot create a BranchPoint for a cell with no values");
        }
    }

    pub fn for_value(step: usize, attribution: Key<WithId>, val: V, cells: Vec<Index>) -> Self {
        if cells.len() > 0 {
            BranchPoint {
                branch_step: step,
                branch_attribution: attribution,
                choices: BranchOver::Value(val, cells, 0),
            }
        } else {
            panic!("Cannot create a BranchPoint for a value with no cells");
        }
    }

    pub fn chosen(&self) -> Option<(Index, V)> {
        match &self.choices {
            BranchOver::Empty => None,
            BranchOver::Cell(c, vs, i) => Some((*c, vs[*i].clone())),
            BranchOver::Value(v, cs, i) => Some((cs[*i], v.clone())),
        }
    }

    pub fn remaining(&self) -> usize {
        match &self.choices {
            BranchOver::Empty => 0,
            BranchOver::Cell(_, vs, i) => vs.len() - 1 - i,
            BranchOver::Value(_, cs, i) => cs.len() - 1 - i,
        }
    }

    pub fn advance(&mut self) -> Option<(Index, V)> {
        match &mut self.choices {
            BranchOver::Empty => None,
            BranchOver::Cell(cell, values, i) => {
                if *i < values.len() - 1 {
                    *i += 1;
                    Some((*cell, values[*i]))
                } else {
                    None
                }
            },
            BranchOver::Value(val, cells, i) => {
                if *i < cells.len() - 1 {
                    *i += 1;
                    Some((cells[*i], val.clone()))
                } else {
                    None
                }
            },
        }
    }

    // Opposite of advance. Returns true if this decision should be re-applied,
    // or false if it should be left off the stack.
    pub fn retreat(&mut self) -> bool {
        match &mut self.choices {
            BranchOver::Empty => false,
            BranchOver::Cell(_, _, i) => {
                if *i == 0 {
                    false
                } else {
                    *i -= 1;
                    true
                }
            },
            BranchOver::Value(_, _, i) => {
                if *i == 0 {
                    false
                } else {
                    *i -= 1;
                    true
                }
            },
        }
    }
}

/// This is a grid of UVSets and FeatureVecs. It is used to represent the
/// not-yet-ruled-out values for each cell in the grid, along with features
/// attached to each cell.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionGrid<V: Value> {
    rows: usize,
    cols: usize,
    grid: Box<[(UVSet<V::U>, FeatureVec<FVMaybeNormed>)]>,
    _marker: PhantomData<V>,
}

impl<V: Value> DecisionGrid<V> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            grid: vec![(empty_set::<V>(), FeatureVec::new()); rows * cols].into_boxed_slice(),
            _marker: PhantomData,
        }
    }

    pub fn full(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            grid: vec![(full_set::<V>(), FeatureVec::new()); rows * cols].into_boxed_slice(),
            _marker: PhantomData,
        }
    }

    pub fn get(&self, index: Index) -> &(UVSet<V::U>, FeatureVec<FVMaybeNormed>) {
        &self.grid[index[0] * self.cols + index[1]]
    }

    pub fn get_mut(&mut self, index: Index) -> &mut (UVSet<V::U>, FeatureVec<FVMaybeNormed>) {
        self.grid.get_mut(index[0] * self.cols + index[1]).unwrap()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Intersects the possible values of this grid with the possible values of
    /// another grid. Also merges the features of the two grids.
    pub fn combine_with(&mut self, other: &DecisionGrid<V>) {
        assert!(self.rows == other.rows && self.cols == other.cols,
                "Cannot combine grids of different sizes");
        for i in 0..self.rows {
            for j in 0..self.cols {
                let a = self.get_mut([i, j]);
                let b = other.get([i, j]);
                a.0.intersect_with(&b.0);
                a.1.extend(&b.1);
            }
        }
    }
}

/// This converts an extracted item from a container a Value, making use of the
/// private API to do so.
pub fn to_value<V: Value>(u: UVal<V::U, UVWrapped>) -> V {
    V::from_uval(u.unwrap())
}

/// The puzzle itself as well as other components can be stateful (i.e., they
/// respond to changes in the grid). The trait provides a default do-nothing
/// implementation so that non-stateful components that are required to be
/// stateful for some reason can be trivially stateful.
pub trait Stateful<V: Value>: {
    fn reset(&mut self) {}
    fn apply(&mut self, index: Index, value: V) -> Result<(), Error> {
        let _ = index;
        let _ = value;
        Ok(())
    }
    fn undo(&mut self, index: Index, value: V) -> Result<(), Error> {
        let _ = index;
        let _ = value;
        Ok(())
    }
}

/// Used to give meaningful layout information to a State.
pub trait Overlay: Clone + Debug {
    type Iter<'a>: Iterator<Item = Index> where Self: 'a;
    fn partition_dimension(&self) -> usize;
    fn n_partitions(&self, dim: usize) -> usize;
    fn partition_size(&self, dim: usize, index: usize) -> usize;
    fn enclosing_partition(&self, index: Index, dim: usize) -> Option<usize>;
    fn enclosing_partitions(&self, index: Index) -> Vec<Option<usize>> {
        (0..self.partition_dimension())
            .map(|dim| self.enclosing_partition(index, dim))
            .collect()
    }
    fn partition_iter(&self, dim: usize, index: usize) -> Self::Iter<'_>;
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        for dim in 0..self.partition_dimension() {
            if self.enclosing_partition(i1, dim) == self.enclosing_partition(i2, dim) {
                return true;
            }
        }
        false
    }
    fn all_mutually_visible(&self, indices: &Vec<Index>) -> bool {
        indices.iter().all(|i| self.mutually_visible(indices[0], *i))
    }
}

/// Trait for representing whatever puzzle is being solved in its current state
/// of being (partially) filled in. Ultimately this is just wrapping a Grid, but
/// it may impose additional meanings on the values of the grid.
pub trait State<V: Value, O: Overlay> where Self: Clone + Debug + Stateful<V> {
    const ROWS: usize;
    const COLS: usize;
    fn get(&self, index: Index) -> Option<V>;
    fn overlay(&self) -> &O;
    fn given_actions(&self) -> Vec<(Index, V)>;
}

#[cfg(any(test, feature = "test-util"))]
pub mod test_util {
    use super::*;

    /// Values for use in testing. More or less an NineStdVal
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct TestVal(pub u8);
    impl Display for TestVal {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl Value for TestVal {
        type U = u8;
        fn parse(s: &str) -> Result<Self, Error> {
            match s.parse::<u8>() {
                Ok(u) => Ok(Self(u)),
                Err(_) => Err(Error::new_const("not a valid u8")),
            }
        }
        fn cardinality() -> usize { 9 }
        fn possibilities() -> Vec<Self> { (1..=9).map(TestVal).collect() }
        fn nth(ord: usize) -> TestVal { TestVal((ord as u8)+1) }
        fn ordinal(&self) -> usize { self.0 as usize - 1 }
        fn from_uval(u: UVal<u8, UVUnwrapped>) -> Self { TestVal(u.value()+1) }
        fn to_uval(self) -> UVal<u8, UVWrapped> { UVal::new(self.0-1) }
    }

    /// Trivial one-dimensional overlay where there's one row (and no columns
    /// or boxes) and everything is in it.
    #[derive(Debug, Clone)]
    pub struct OneDimOverlay<const N: usize>;
    impl <const N: usize> Overlay for OneDimOverlay<N> {
        type Iter<'a> = std::vec::IntoIter<Index>;
        fn partition_dimension(&self) -> usize { 1 }
        fn n_partitions(&self, _: usize) -> usize { 1 }
        fn partition_size(&self, _: usize, _: usize) -> usize { 1 }
        fn mutually_visible(&self, _: Index, _: Index) -> bool { true }
        fn enclosing_partition(&self, _: Index, dim: usize) -> Option<usize> {
            assert_eq!(dim, 0);
            Some(0)
        }
        fn enclosing_partitions(&self, _: Index) -> Vec<Option<usize>> { vec![Some(0)] }
        fn partition_iter(&self, dim: usize, index: usize) -> Self::Iter<'_> {
            assert_eq!(dim, 0);
            assert_eq!(index, 0);
            (0..N).map(|x| [0, x]).collect::<Vec<Index>>().into_iter()
        }
    }

    /// Trivial 1-D grid.
    #[derive(Debug, Clone)]
    pub struct OneDim<const N: usize> {
        pub grid: UVGrid<u8>,
    }
    impl <const N: usize> OneDim<N> {
        pub fn new() -> Self { Self { grid: UVGrid::new(1, N) } }
        pub fn full_dg() -> DecisionGrid<TestVal> { DecisionGrid::full(1, N) }
        pub fn to_string(&self) -> String {
            (0..N).map(|i| {
                if let Some(v) = self.get([0, i]) {
                    format!("{}", v.0)
                } else {
                    ".".to_string()
                }
            }).collect::<Vec<_>>().join("")
        }
    }
    impl <const N: usize> Stateful<TestVal> for OneDim<N> {
        fn reset(&mut self) { self.grid = UVGrid::new(Self::ROWS, Self::COLS); }
        fn apply(&mut self, index: Index, value: TestVal) -> Result<(), Error> {
            self.grid.set(index, Some(value.to_uval()));
            Ok(())
        }
        fn undo(&mut self, index: Index, _: TestVal) -> Result<(), Error> {
            self.grid.set(index, None);
            Ok(())
        }
    }
    impl <const N: usize> State<TestVal, OneDimOverlay<N>> for OneDim<N> {
        const ROWS: usize = 1;
        const COLS: usize = N;
        fn get(&self, index: Index) -> Option<TestVal> { self.grid.get(index).map(to_value) }
        fn overlay(&self) -> &OneDimOverlay<N> { &OneDimOverlay{} }
        fn given_actions(&self) -> Vec<(Index, TestVal)> { vec![] }
    }

    /// Unwrapping UVals is private to the core module, but it's valuable to
    /// check that the to_uval/from_uval methods successfully round-trip values.
    pub fn round_trip_value<V: Value>(v: V) -> V {
        let u: UVal<V::U, UVWrapped> = v.to_uval();
        V::from_uval(u.unwrap())
    }

    /// Most of the time, you can just rely on the solver to replay given
    /// actions, but for tests, you may want to parse a state and check that
    /// the givens are right.
    pub fn replay_givens<V: Value, O: Overlay, S: State<V, O>>(state: &mut S) {
        for (i, v) in state.given_actions() {
            state.apply(i, v).unwrap();
        }
    }
}