use std::collections::HashMap;
use std::sync::Mutex;
use std::{borrow::Cow, marker::PhantomData};
use std::fmt::Debug;
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
pub trait Value<U: UInt>: Copy + Clone + Debug + PartialEq + Eq {
    fn cardinality() -> usize;
    fn possibilities() -> Vec<Self>;
    fn parse(s: &str) -> Result<Self, Error>;

    fn from_uval(u: UVal<U, UVUnwrapped>) -> Self;
    fn to_uval(self) -> UVal<U, UVWrapped>;
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
#[derive(Debug)]
pub struct UVMap<U: UInt, V: Clone> {
    vals: Box<[V]>,
    _p_u: PhantomData<U>,
}

pub fn empty_map<U: UInt, K: Value<U>, V: Clone + Default>() -> UVMap<U, V> {
    UVMap { vals: vec![V::default(); K::cardinality()].into_boxed_slice(), _p_u: PhantomData }
}

pub fn filled_map<U: UInt, K: Value<U>, V: Clone>(default: V) -> UVMap<U, V> {
    UVMap { vals: vec![default; K::cardinality()].into_boxed_slice(), _p_u: PhantomData }
}

impl <U: UInt, V: Clone> UVMap<U, V> {
    pub fn get(&self, key: UVal<U, UVWrapped>) -> &V {
        &self.vals[key.unwrap().value().as_usize()]
    }

    pub fn get_mut(&mut self, key: UVal<U, UVWrapped>) -> &mut V {
        &mut self.vals[key.unwrap().value().as_usize()]
    }

    pub fn iter<K: Value<U>>(&self) -> std::vec::IntoIter<(K, V)> {
        self.vals.iter().enumerate().map(|(u, v)| {
            let k: K = to_value::<U, K>(UVal::<U, UVWrapped>::new(U::from_usize(u)));
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
    _p_u: PhantomData<U>,
}

pub fn empty_set<U: UInt, V: Value<U>>() -> UVSet<U> {
    UVSet {
        s: BitSet::with_capacity(V::cardinality()),
        _p_u: PhantomData,
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

pub fn full_set<U: UInt, V: Value<U>>() -> UVSet<U> {
    let n = V::cardinality();
    let mut s = UVSet {
        s: BitSet::with_capacity(n),
        _p_u: PhantomData,
    };
    let ones = leading_ones(n);
    s.s.union_with(&BitSet::from_bytes(ones.as_slice()));
    s
}

pub fn pack_values<U: UInt, V: Value<U>>(vals: &Vec<V>) -> UVSet<U> {
    let mut res = empty_set::<U, V>();
    for v in vals {
        res.insert(v.to_uval());
    }
    res
}

pub fn singleton_set<U: UInt, V: Value<U>>(v: V) -> UVSet<U> {
    let mut s = empty_set::<U, V>();
    s.insert(v.to_uval());
    s
}

pub fn unpack_values<U: UInt, V: Value<U>>(s: &UVSet<U>) -> Vec<V> {
    s.iter().map(|u| { to_value::<U, V>(u) }).collect::<Vec<_>>()
}

pub fn unpack_singleton<U: UInt, V: Value<U>>(s: &UVSet<U>) -> Option<V> {
    if s.len() == 1 {
        Some(to_value::<U, V>(s.iter().next().unwrap()))
    } else {
        None
    }
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
    pub fn get_name(&self, id: usize) -> Option<&'static str> {
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
    static ref ATTRIBUTION_REGISTRY: Mutex<ConstStringRegistry> = {
        Mutex::new(ConstStringRegistry::new())
    };
}

// NOTE: This is an expensive operation, so only use it for human-interface
// purposes (e.g., debugging, logging, etc.) and not during the solving process.
pub fn readable_attribution(id: usize) -> Option<Attribution<WithId>> {
    let registry = ATTRIBUTION_REGISTRY.lock().unwrap();
    registry.get_name(id).map(|name| {
        Attribution { name, id: Some(id), _state: PhantomData}
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attribution<S>{
    name: &'static str,
    id: Option<usize>,
    _state: PhantomData<S>,
}

impl <S> Attribution<S> {
    pub fn get_name(&self) -> &'static str { self.name }
}

impl Attribution<MaybeId> {
    // Attributions are lazily initialized; the id is set when it is first used.
    pub fn new(name: &'static str) -> Self {
        Attribution { name, id: None, _state: PhantomData }
    }

    pub fn unwrap(&mut self) -> Attribution<WithId> {
        if let Some(id) = self.id {
            return Attribution { name: self.name, id: Some(id), _state: PhantomData };
        } else {
            let id = ATTRIBUTION_REGISTRY.lock().unwrap().register(self.name);
            self.id = Some(id);
            return Attribution { name: self.name, id: Some(id), _state: PhantomData };
        }
    }
}

impl Attribution<WithId> {
    pub fn get_id(&self) -> usize { self.id.unwrap() }
}

lazy_static::lazy_static! {
    static ref FEATURE_REGISTRY: Mutex<ConstStringRegistry> = {
        Mutex::new(ConstStringRegistry::new())
    };
}

// NOTE: This is an expensive operation, so only use it for human-interface
// purposes (e.g., debugging, logging, etc.) and not during the solving process.
pub fn readable_feature(id: usize) -> Option<FeatureKey<WithId>> {
    let registry = FEATURE_REGISTRY.lock().unwrap();
    registry.get_name(id).map(|name| {
        FeatureKey { name, id: Some(id), _state: PhantomData}
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureKey<S>{
    name: &'static str,
    id: Option<usize>,
    _state: PhantomData<S>,
}

impl <S> FeatureKey<S> {
    pub fn get_name(&self) -> &'static str { self.name }
}

impl FeatureKey<MaybeId> {
    // Features are lazily initialized; the id is set when it is first used.
    pub fn new(name: &'static str) -> Self {
        FeatureKey { name, id: None, _state: PhantomData }
    }

    pub fn unwrap(&mut self) -> FeatureKey<WithId> {
        if let Some(id) = self.id {
            return FeatureKey { name: self.name, id: Some(id), _state: PhantomData };
        } else {
            let id = FEATURE_REGISTRY.lock().unwrap().register(self.name);
            self.id = Some(id);
            return FeatureKey { name: self.name, id: Some(id), _state: PhantomData };
        }
    }
}

impl FeatureKey<WithId> {
    pub fn get_id(&self) -> usize { self.id.unwrap() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FVMaybeNormed;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FVNormed;

#[derive(Clone, PartialEq)]
pub struct FeatureVec<S> {
    features: Vec<(usize, f64)>,
    // Some methods only make sense when the features are sorted by id and
    // duplicate ids are dropped or merged.
    normalized: bool,
    _p_s: PhantomData<S>,
}

impl FeatureVec<FVMaybeNormed> {
    pub fn new() -> Self {
        FeatureVec { features: Vec::new(), normalized: true, _p_s: PhantomData }
    }

    pub fn from_pairs(weights: Vec<(&'static str, f64)>) -> Self {
        let mut fv = FeatureVec::new();
        for (k, v) in weights {
            fv.add(&FeatureKey::new(k).unwrap(), v);
        }
        fv
    }

    pub fn add(&mut self, key: &FeatureKey<WithId>, value: f64) {
        let id = key.get_id();
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

impl Debug for FeatureVec<FVMaybeNormed> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for (i, (k, v)) in self.features.iter().enumerate() {
            write!(f, "{:?} => {}", readable_feature(*k), *v)?;
            if i+1 < self.features.len() {
                write!(f, ", ")?;
            }
        }
        write!(f, "}}")
    }
}

impl FeatureVec<FVNormed> {
    pub fn get(&self, key: &FeatureKey<WithId>) -> Option<f64> {
        let id = key.get_id();
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
pub struct CertainDecision<U: UInt, V: Value<U>> {
    pub index: Index,
    pub value: V,
    _p_u: PhantomData<U>,
}

impl <U: UInt, V: Value<U>> CertainDecision<U, V> {
    pub fn new(index: Index, value: V) -> Self {
        Self { index, value, _p_u: PhantomData }
    }
}

/// Constraints and ranking both may return early if they hit upon either a
/// contradiction or a certainty. This is a simple enum to represent this
/// short-circuiting.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintResult<U: UInt, V: Value<U>> {
    Contradiction(Attribution<WithId>),
    Certainty(CertainDecision<U, V>, Attribution<WithId>),
    Ok,
}

impl <U: UInt, V: Value<U>> ConstraintResult<U, V> {
    pub fn is_ok(&self) -> bool {
        match self {
            ConstraintResult::Ok => true,
            _ => false,
        }
    }
}

/// When choosing to branch, we can either try all the possible values for a
/// particular cell, or we can try all possible cells for a particular value.
#[derive(Debug, Clone)]
pub enum BranchOver<U: UInt, S: State<U>> {
    Empty,
    Cell(Index, std::vec::IntoIter<S::Value>),
    Value(S::Value, std::vec::IntoIter<Index>),
}

/// A decision point in the puzzle. This includes the specific value that was
/// chosen, as well as the index of the cell that was modified, as well as the
/// alternative values/indices that have not been tried yet.
#[derive(Debug, Clone)]
pub struct BranchPoint<U: UInt, S: State<U>> {
    pub chosen: Option<(Index, S::Value)>,
    pub chosen_step: usize,
    pub alternatives: BranchOver<U, S>,
}

impl <U: UInt, S: State<U>> BranchPoint<U, S> {
    pub fn unique(step: usize, index: Index, value: S::Value) -> Self {
        BranchPoint {
            chosen: Some((index, value)),
            chosen_step: step,
            alternatives: BranchOver::Empty,
        }
    }

    pub fn empty(step: usize) -> Self {
        BranchPoint { chosen: None, chosen_step: step, alternatives: BranchOver::Empty }
    }

    pub fn for_cell(step: usize, index: Index, alternatives: Vec<S::Value>) -> Self {
        let mut iter = alternatives.into_iter();
        if iter.len() > 0 {
            BranchPoint {
                chosen: Some((index, iter.next().unwrap())),
                chosen_step: step,
                alternatives: BranchOver::Cell(index, iter),
            }
        } else {
            panic!("Cannot create a BranchPoint for a cell with no alternatives");
        }
    }

    pub fn for_value(step: usize, val: S::Value, alternatives: Vec<Index>) -> Self {
        let mut iter = alternatives.into_iter();
        if iter.len() > 0 {
            BranchPoint {
                chosen: Some((iter.next().unwrap(), val)),
                chosen_step: step,
                alternatives: BranchOver::Value(val, iter),
            }
        } else {
            panic!("Cannot create a BranchPoint for a value with no alternatives");
        }
    }

    pub fn is_empty(&self) -> bool {
        self.chosen.is_none() && match &self.alternatives {
            BranchOver::Empty => true,
            BranchOver::Cell(_, iter) => iter.len() == 0,
            BranchOver::Value(_, iter) => iter.len() == 0,
        }
    }

    pub fn len(&self) -> usize {
        match &self.alternatives {
            BranchOver::Empty => 0,
            BranchOver::Cell(_, iter) => iter.len(),
            BranchOver::Value(_, iter) => iter.len(),
        }
    }

    pub fn advance(&mut self) -> Option<(Index, S::Value)> {
        match &mut self.alternatives {
            BranchOver::Empty => None,
            BranchOver::Cell(index, iter) => {
                if let Some(next) = iter.next() {
                    self.chosen = Some((*index, next));
                } else {
                    self.chosen = None;
                    self.alternatives = BranchOver::Empty;
                }
                self.chosen
            },
            BranchOver::Value(val, iter) => {
                if let Some(next) = iter.next() {
                    self.chosen = Some((next, *val));
                } else {
                    self.chosen = None;
                    self.alternatives = BranchOver::Empty;
                }
                self.chosen
            },
        }
    }
}

/// This is a grid of UVSets and FeatureVecs. It is used to represent the
/// not-yet-ruled-out values for each cell in the grid, along with features
/// attached to each cell.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionGrid<U: UInt, V: Value<U>> {
    rows: usize,
    cols: usize,
    grid: Box<[(UVSet<U>, FeatureVec<FVMaybeNormed>)]>,
    _p_v: PhantomData<V>,
}

impl<U: UInt, V: Value<U>> DecisionGrid<U, V> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            grid: vec![(empty_set::<U, V>(), FeatureVec::new()); rows * cols].into_boxed_slice(),
            _p_v: PhantomData,
        }
    }

    pub fn full(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            grid: vec![(full_set::<U, V>(), FeatureVec::new()); rows * cols].into_boxed_slice(),
            _p_v: PhantomData,
        }
    }

    pub fn get(&self, index: Index) -> &(UVSet<U>, FeatureVec<FVMaybeNormed>) {
        &self.grid[index[0] * self.cols + index[1]]
    }

    pub fn get_mut(&mut self, index: Index) -> &mut (UVSet<U>, FeatureVec<FVMaybeNormed>) {
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
    pub fn combine_with(&mut self, other: &DecisionGrid<U, V>) {
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
pub fn to_value<U: UInt, V: Value<U>>(u: UVal<U, UVWrapped>) -> V {
    V::from_uval(u.unwrap())
}

/// The puzzle itself as well as other components can be stateful (i.e., they
/// respond to changes in the grid). The trait provides a default do-nothing
/// implementation so that non-stateful components that are required to be
/// stateful for some reason can be trivially stateful.
pub trait Stateful<U: UInt, V: Value<U>>: {
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

/// Trait for representing whatever puzzle is being solved in its current state
/// of being (partially) filled in. Ultimately this is just wrapping a Grid, but
/// it may impose additional meanings on the values of the grid.
pub trait State<U: UInt> where Self: Clone + Debug + Stateful<U, Self::Value> {
    type Value: Value<U>;
    const ROWS: usize;
    const COLS: usize;
    fn get(&self, index: Index) -> Option<Self::Value>;
}

#[cfg(test)]
pub mod test_util {
    use super::*;
    /// Unwrapping UVals is private to the core module, but it's valuable to
    /// check that the to_uval/from_uval methods successfully round-trip values.
    pub fn round_trip_value<U: UInt, V: Value<U>>(v: V) -> V {
        let u: UVal<U, UVWrapped> = v.to_uval();
        V::from_uval(u.unwrap())
    }
}