use std::collections::HashMap;
use std::sync::Mutex;
use std::{borrow::Cow, marker::PhantomData};
use std::fmt::{Debug, Display};
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
#[derive(Clone)]
pub struct VGrid<V: Value> {
    rows: usize,
    cols: usize,
    grid: Box<[Option<V::U>]>,
    _marker: PhantomData<V>,
}

impl <V: Value> Debug for VGrid<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}: ", self.rows, self.cols)?;
        for r in 0..self.rows {
            write!(f, "[")?;
            for c in 0..self.cols {
                if let Some(u) = self.grid[r * self.cols + c] {
                    write!(f, "{}", to_value::<V>(UVal::new(u)))?;
                } else {
                    write!(f, "_")?;
                }
                if c != self.cols - 1 {
                    write!(f, ",")?;
                }
            }
            write!(f, "[")?;
        }
        Ok(())
    }
}

impl<V: Value> VGrid<V> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            grid: vec![None; rows * cols].into_boxed_slice(),
            _marker: PhantomData,
        }
    }

    pub fn get(&self, index: Index) -> Option<V> {
        self.grid[index[0] * self.cols + index[1]].map(|v| to_value::<V>(UVal::new(v)))
    }

    pub fn set(&mut self, index: Index, value: Option<V>) {
        self.grid[index[0] * self.cols + index[1]] = value.map(|v| v.to_uval().unwrap().value());
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// These exist in order to abstract over VDenseMaps and ref-storing versions of
/// the same.
pub trait DenseMapState<V: Clone> {
    fn vals(&self) -> &Box<[V]>;
}
pub trait DenseMapStateMut<V: Clone>: DenseMapState<V> {
    fn vals_mut(&mut self) -> &mut Box<[V]>;
}

/// This is the primary read-only functionality for manipulating maps from
/// Values to some other type, backed by a Box<[V]> or a reference to one.
pub trait VMap<K: Value, V: Clone>: DenseMapState<V> {
    fn get(&self, key: &K) -> &V {
        &self.vals()[key.to_uval().unwrap().value().as_usize()]
    }

    fn iter(&self) -> std::vec::IntoIter<(K, V)> {
        self.vals().iter().enumerate().map(|(u, v)| {
            let k: K = to_value::<K>(UVal::<K::U, UVWrapped>::new(K::U::from_usize(u)));
            let v: V = v.clone();
            (k, v)
        }).collect::<Vec<(K, V)>>().into_iter()
    }
}

/// This is the primary functionality to modify maps from Values to some other
/// type, backed by a bitset or a reference to one.
pub trait VMapMut<K: Value, V: Clone>: DenseMapStateMut<V> {
    fn get_mut(&mut self, key: &K) -> &mut V {
        &mut self.vals_mut()[key.to_uval().unwrap().value().as_usize()]
    }
}

/// This is a mapping from values to some other type. It is densely represented
/// as a slice of values, indexed by their unsigned representations.
#[derive(Debug, Clone)]
pub struct VDenseMap<K: Value, V: Clone> {
    vals: Box<[V]>,
    _marker: PhantomData<K>,
}

impl <K: Value, V: Clone + Default> VDenseMap<K, V> {
    pub fn empty() -> Self {
        Self { vals: vec![V::default(); K::cardinality()].into_boxed_slice(), _marker: PhantomData }
    }
}

impl <K: Value, V: Clone> VDenseMap<K, V> {
    pub fn filled(default: V) -> Self {
        Self { vals: vec![default; K::cardinality()].into_boxed_slice(), _marker: PhantomData }
    }

    pub fn into_erased(self) -> Box<[V]> {
        self.vals
    }
}

impl <K: Value, V: Clone> DenseMapState<V> for VDenseMap<K, V> {
    fn vals(&self) -> &Box<[V]> { &self.vals }
}
impl <K: Value, V: Clone> DenseMapStateMut<V> for VDenseMap<K, V> {
    fn vals_mut(&mut self) -> &mut Box<[V]> { &mut self.vals }
}
impl <K: Value, V: Clone> VMap<K, V> for VDenseMap<K, V> {}
impl <K: Value, V: Clone> VMapMut<K, V> for VDenseMap<K, V> {}

#[derive(Debug)]
pub struct VDenseMapRefMut<'a, K: Value, V: Clone> {
    vals: &'a mut Box<[V]>,
    _marker: PhantomData<K>,
}

impl <'a, K: Value, V: Clone> VDenseMapRefMut<'a, K, V> {
    /// Warning: Strictly speaking, this is not "unsafe", but there's no
    /// guarantees about the contents of the VMap you get if the K/V you assume
    /// differs from the ones used to create it.
    pub fn assume_typed(vals: &'a mut Box<[V]>) -> Self {
        Self { vals, _marker: PhantomData }
    }
    pub fn get_into(self, key: &K) -> &'a mut V {
        self.vals.get_mut(key.to_uval().unwrap().value().as_usize()).unwrap()
    }
    pub fn to_vdensemap(&self) -> VDenseMap<K, V> {
        VDenseMap { vals: self.vals.clone(), _marker: PhantomData }
    }
}
impl <'a, K: Value, V: Clone> DenseMapState<V> for VDenseMapRefMut<'a, K, V> {
    fn vals(&self) -> &Box<[V]> { self.vals }
}
impl <'a, K: Value, V: Clone> DenseMapStateMut<V> for VDenseMapRefMut<'a, K, V> {
    fn vals_mut(&mut self) -> &mut Box<[V]> { self.vals }
}
impl <'a, K: Value, V: Clone> VMap<K, V> for VDenseMapRefMut<'a, K, V> {}
impl <'a, K: Value, V: Clone> VMapMut<K, V> for VDenseMapRefMut<'a, K, V> {}

#[derive(Debug)]
pub struct VDenseMapRef<'a, K: Value, V: Clone> {
    vals: &'a Box<[V]>,
    _marker: PhantomData<K>,
}

impl <'a, K: Value, V: Clone> VDenseMapRef<'a, K, V> {
    /// Warning: Strictly speaking, this is not "unsafe", but there's no
    /// guarantees about the contents of the VMap you get if the K/V you assume
    /// differs from the ones used to create it.
    pub fn assume_typed(vals: &'a Box<[V]>) -> Self {
        Self { vals, _marker: PhantomData }
    }
    pub fn get_into(self, key: &K) -> &'a V {
        &self.vals[key.to_uval().unwrap().value().as_usize()]
    }
    pub fn to_vdensemap(&self) -> VDenseMap<K, V> {
        VDenseMap { vals: self.vals.clone(), _marker: PhantomData }
    }
}
impl <'a, K: Value, V: Clone> DenseMapState<V> for VDenseMapRef<'a, K, V> {
    fn vals(&self) -> &Box<[V]> { self.vals }
}
impl <'a, K: Value, V: Clone> VMap<K, V> for VDenseMapRef<'a, K, V> {}

/// These exist in order to abstract over VBitSets and ref-storing versions of
/// the same.
pub trait BitSetState<V: Value> {
    fn s(&self) -> &bit_set::BitSet;
}
pub trait BitSetStateMut<V: Value>: BitSetState<V> {
    fn s_mut(&mut self) -> &mut bit_set::BitSet;
}

/// This is the primary read-only functionality for manipulating sets of Values,
/// backed by a bitset or a reference to one.
pub trait VSet<V: Value>: BitSetState<V> {
    fn contains(&self, value: &V) -> bool {
        self.s().contains(value.to_uval().unwrap().value().as_usize())
    }

    fn is_empty(&self) -> bool {
        self.s().is_empty()
    }

    fn len(&self) -> usize {
        self.s().len()
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = V> + 'a {
        self.s().iter().map(|i| to_value::<V>(UVal::new(V::U::from_usize(i))))
    }

    fn intersection<O: VSet<V>>(&self, other: &O) -> VBitSet<V> {
        let mut s = self.s().clone();
        s.intersect_with(other.s());
        VBitSet { s, _marker: PhantomData }
    }

    fn union<O: VSet<V>>(&self, other: &O) -> VBitSet<V> {
        let mut s = self.s().clone();
        s.union_with(other.s());
        VBitSet { s, _marker: PhantomData }
    }

    fn first(&self) -> Option<V> {
        self.iter().next()
    }

    fn last(&self) -> Option<V> {
        self.iter().last()
    }

    fn as_singleton(&self) -> Option<V> {
        if self.len() == 1 {
            Some(self.iter().next().unwrap())
        } else {
            None
        }
    }

    fn to_string(&self) -> String {
        let mut s = "[".to_string();
        let len = self.len();
        for (i, v) in self.iter().enumerate() {
            s.push_str(format!("{}", v).as_str());
            if i+1 < len {
                s.push_str(", ");
            }
        }
        s.push(']');
        s
    }
}

/// This is the primary functionality to modify sets of Values, backed by a
/// bitset or a reference to one.
pub trait VSetMut<V: Value>: BitSetStateMut<V> {
    fn remove(&mut self, value: &V) {
        self.s_mut().remove(value.to_uval().unwrap().value().as_usize());
    }

    fn clear(&mut self) {
        self.s_mut().clear();
    }

    fn intersect_with<O: VSet<V>>(&mut self, other: &O) {
        self.s_mut().intersect_with(other.s());
    }
}

/// This a set of values (e.g., that are possible, that have been seen, etc.).
/// They are represented as a bitset of the possible values.
#[derive(Debug, Clone, PartialEq)]
pub struct VBitSet<V: Value> {
    s: bit_set::BitSet,
    _marker: PhantomData<V>,
}

/// Constructors are specific to VBitSet, all the other methods
/// are general across VBitSet and VBitSetRef (thorugh the VSet trait).
impl <V: Value> VBitSet<V> {
    pub fn empty() -> Self {
        Self {
            s: bit_set::BitSet::with_capacity(V::cardinality()),
            _marker: PhantomData,
        }
    }

    pub fn full() -> Self {
        let n = V::cardinality();
        let mut s = VBitSet {
            s: bit_set::BitSet::with_capacity(n),
            _marker: PhantomData,
        };
        let ones = leading_ones(n);
        s.s.union_with(&bit_set::BitSet::from_bytes(ones.as_slice()));
        s
    }

    pub fn from_values(vals: &Vec<V>) -> Self {
        let mut res = Self::empty();
        for v in vals {
            res.insert(v);
        }
        res
    }

    pub fn singleton(v: &V) -> Self {
        let mut s = Self::empty();
        s.insert(v);
        s
    }

    pub fn insert(&mut self, value: &V) {
        self.s.insert(value.to_uval().unwrap().value().as_usize());
    }

    pub fn into_erased(self) -> bit_set::BitSet {
        self.s
    }
}

impl <V: Value> BitSetState<V> for VBitSet<V> {
    fn s(&self) -> &bit_set::BitSet { &self.s }
}
impl <V: Value> BitSetStateMut<V> for VBitSet<V> {
    fn s_mut(&mut self) -> &mut bit_set::BitSet { &mut self.s }
}
impl <V: Value> VSet<V> for VBitSet<V> {}
impl <V: Value> VSetMut<V> for VBitSet<V> {}

#[derive(Debug, PartialEq)]
pub struct VBitSetRefMut<'a, V: Value> {
    s: &'a mut bit_set::BitSet,
    _marker: PhantomData<V>,
}

impl <'a, V: Value> VBitSetRefMut<'a, V> {
    /// Warning: Strictly speaking, this is not "unsafe", but there's no
    /// guarantees about the contents of the VSet you get if the V you assume
    /// differs from the one used to create it.
    pub fn assume_typed(s: &'a mut bit_set::BitSet) -> Self {
        Self { s, _marker: PhantomData }
    }
    pub fn to_vbitset(&self) -> VBitSet<V> {
        VBitSet { s: self.s.clone(), _marker: PhantomData }
    }
}

impl <'a, V: Value> BitSetState<V> for VBitSetRefMut<'a, V> {
    fn s(&self) -> &bit_set::BitSet { self.s }
}
impl <'a, V: Value> BitSetStateMut<V> for VBitSetRefMut<'a, V> {
    fn s_mut(&mut self) -> &mut bit_set::BitSet { self.s }
}
impl <'a, V: Value> VSet<V> for VBitSetRefMut<'a, V> {}
impl <'a, V: Value> VSetMut<V> for VBitSetRefMut<'a, V> {}

#[derive(Debug, PartialEq)]
pub struct VBitSetRef<'a, V: Value> {
    s: &'a bit_set::BitSet,
    _marker: PhantomData<V>,
}

impl <'a, V: Value> VBitSetRef<'a, V> {
    /// Warning: Strictly speaking, this is not "unsafe", but there's no
    /// guarantees about the contents of the VSet you get if the V you assume
    /// differs from the one used to create it.
    pub fn assume_typed(s: &'a bit_set::BitSet) -> Self {
        Self { s, _marker: PhantomData }
    }
    pub fn to_vbitset(&self) -> VBitSet<V> {
        VBitSet { s: self.s.clone(), _marker: PhantomData }
    }
}

impl <'a, V: Value> BitSetState<V> for VBitSetRef<'a, V> {
    fn s(&self) -> &bit_set::BitSet { self.s }
}
impl <'a, V: Value> VSet<V> for VBitSetRef<'a, V> {}

fn leading_ones(n: usize) -> Vec<u8> {
    let full = n / 8;
    let remaining = n % 8;
    let mut result = vec![u8::MAX; full];
    if remaining > 0 {
        result.push(u8::MAX << (8 - remaining));
    }
    result
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

/// Types of Keys
pub trait KeyType: 'static + Debug + Clone + Copy + PartialEq + Eq {
    fn type_id() -> u8;
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attribution;
impl KeyType for Attribution { fn type_id() -> u8 { 1 } }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Feature;
impl KeyType for Feature { fn type_id() -> u8 { 2 } }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionLayer;
impl KeyType for RegionLayer { fn type_id() -> u8 { 3 } }

pub const ROWS_LAYER: Key<RegionLayer> = Key {
    name: "ROWS", id: 0, _state: PhantomData,
};
pub const COLS_LAYER: Key<RegionLayer> = Key {
    name: "COLS", id: 1, _state: PhantomData,
};
pub const BOXES_LAYER: Key<RegionLayer> = Key {
    name: "BOXES", id: 2, _state: PhantomData,
};

lazy_static::lazy_static! {
    static ref KEY_REGISTRY: Mutex<HashMap<u8, ConstStringRegistry>> = {
        let mut h = HashMap::new();
        h.insert(Attribution::type_id(), ConstStringRegistry::new());
        h.insert(Feature::type_id(), ConstStringRegistry::new());
        let mut p = ConstStringRegistry::new();
        // These partition types have special treatment so they can serve as
        // indices without hashing or searching.
        assert_eq!(p.register("ROWS"), 0);
        assert_eq!(p.register("COLS"), 1);
        assert_eq!(p.register("BOXES"), 2);
        h.insert(RegionLayer::type_id(), p);
        Mutex::new(h)
    };
}

// NOTE: This is an expensive operation, so only use it for human-interface
// purposes (e.g., debugging, logging, etc.) and not during the solving process.
pub fn readable_key<KT: KeyType>(id: usize) -> Option<Key<KT>> {
    let registry = KEY_REGISTRY.lock().unwrap();
    registry.get(&KT::type_id()).unwrap().name(id).map(|name| {
        Key { name, id, _state: PhantomData}
    })
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Key<KT: KeyType>{
    name: &'static str,
    id: usize,
    _state: PhantomData<KT>,
}

impl <KT: KeyType> Key<KT> {
    pub fn register(name: &'static str) -> Self {
        let type_id = KT::type_id();
        let mut kr = KEY_REGISTRY.lock().unwrap();
        let id = kr.get_mut(&type_id).unwrap().register(name);
        return Key { name, id, _state: PhantomData };
    }
    pub fn name(&self) -> &'static str { self.name }
    pub fn id(&self) -> usize { self.id }
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
    Contradiction(Key<Attribution>),
    Certainty(CertainDecision<V>, Key<Attribution>),
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
    pub branch_attribution: Key<Attribution>,
    pub choices: BranchOver<V>,
}

impl <V: Value> BranchPoint<V> {
    pub fn unique(step: usize, attribution: Key<Attribution>, index: Index, value: V) -> Self {
        Self::for_cell(step, attribution, index, vec![value])
    }

    pub fn empty(step: usize, attribution: Key<Attribution>) -> Self {
        BranchPoint { branch_step: step, branch_attribution: attribution, choices: BranchOver::Empty }
    }

    pub fn for_cell(step: usize, attribution: Key<Attribution>, index: Index, values: Vec<V>) -> Self {
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

    pub fn for_value(step: usize, attribution: Key<Attribution>, val: V, cells: Vec<Index>) -> Self {
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
    fn grid_dims(&self) -> (usize, usize);
    fn region_layers(&self) -> Vec<Key<RegionLayer>>;
    fn regions_in_layer(&self, layer: Key<RegionLayer>) -> usize;
    fn cells_in_region(&self, layer: Key<RegionLayer>, index: usize) -> usize;
    fn enclosing_region_and_offset(&self, layer: Key<RegionLayer>, index: Index) -> Option<(usize, usize)>;
    fn nth_in_region(&self, layer: Key<RegionLayer>, index: usize, offset: usize) -> Option<Index>;
    fn region_iter(&self, layer: Key<RegionLayer>, index: usize) -> Self::Iter<'_>;
    fn mutually_visible(&self, i1: Index, i2: Index) -> bool {
        for layer in self.region_layers() {
            if let Some((r1, _)) = self.enclosing_region_and_offset(layer, i1) {
                if let Some((r2, _)) = self.enclosing_region_and_offset(layer, i2) {
                    return r1 == r2
                } else {
                    return false
                }
            } else {
                return self.enclosing_region_and_offset(layer, i2).is_none();
            }
        }
        false
    }
    fn all_mutually_visible(&self, indices: &Vec<Index>) -> bool {
        indices.iter().all(|i| self.mutually_visible(indices[0], *i))
    }
    fn parse_state<V: Value>(&self, s: &str) -> Result<State<V, Self>, Error>;
    fn serialize_state<V: Value>(&self, s: &State<V, Self>) -> String;
}

/// The state of the puzzle being solved (whether empty, full, or partially
/// filled in). It is ultimately just a grid of Values with the interpretations
/// being determined by the Constraints, but the associated Overlay can provide
/// some global structure (layers, rows/cols/boxes, regions).
#[derive(Clone)]
pub struct State<V: Value, O: Overlay> {
    n: usize,
    m: usize,
    grid: VGrid<V>,
    given: VGrid<V>,
    overlay: O,
}

pub const DIMENSION_MISMATCH_ERROR: Error = Error::new_const("Mismatched grid dimensions");
pub const OUT_OF_BOUNDS_ERROR: Error = Error::new_const("Out of bounds");
pub const ALREADY_FILLED_ERROR: Error = Error::new_const("Cell already filled");
pub const NO_SUCH_ACTION_ERROR: Error = Error::new_const("No such action to undo");
pub const UNDO_MISMATCH_ERROR: Error = Error::new_const("Undo value mismatch");

impl <V: Value, O: Overlay> State<V, O> {
    pub fn new(overlay: O) -> Self {
        let (n, m) = overlay.grid_dims();
        Self {
            n, m,
            grid: VGrid::new(n, m),
            given: VGrid::new(n, m),
            overlay,
        }
    }

    pub fn with_givens(overlay: O, given: VGrid<V>) -> Result<Self, Error> {
        let (n, m) = overlay.grid_dims();
        if given.rows() != n || given.cols() != m {
            return Err(DIMENSION_MISMATCH_ERROR);
        }
        Ok(Self {
            n, m,
            grid: VGrid::new(n, m),
            given,
            overlay,
        })
    }

    pub fn get(&self, index: Index) -> Option<V> {
        if index[0] >= self.n || index[1] >= self.m {
            return None;
        }
        self.grid.get(index)
    }

    pub fn overlay(&self) -> &O { &self.overlay }

    pub fn given_actions(&self) -> Vec<(Index, V)> {
        (0..self.n).flat_map(|r| {
            (0..self.m).filter_map(move |c| {
                self.given.get([r, c]).map(|v| {
                    ([r, c], v)
                })
            })
        }).collect()
    }
}

impl <V: Value, O: Overlay> Debug for State<V, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.overlay.serialize_state(&self))
    }
}

impl <V: Value, O: Overlay> Stateful<V> for State<V, O> {
    fn reset(&mut self) {
        self.grid = VGrid::new(self.n, self.m);
    }

    fn apply(&mut self, index: Index, value: V) -> Result<(), Error> {
        if index[0] >= self.n || index[1] >= self.m {
            return Err(OUT_OF_BOUNDS_ERROR);
        }
        if self.grid.get(index).is_some() {
            return Err(ALREADY_FILLED_ERROR);
        }
        self.grid.set(index, Some(value));
        Ok(())
    }

    fn undo(&mut self, index: Index, value: V) -> Result<(), Error> {
        if index[0] >= self.n || index[1] >= self.m {
            return Err(OUT_OF_BOUNDS_ERROR);
        }
        match self.grid.get(index) {
            None => return Err(NO_SUCH_ACTION_ERROR),
            Some(v) => {
                if v != value {
                    return Err(UNDO_MISMATCH_ERROR);
                }
            }
        }
        self.grid.set(index, None);
        Ok(())
    }
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
        fn grid_dims(&self) -> (usize, usize) { (1, N) }
        fn region_layers(&self) -> Vec<Key<RegionLayer>> { vec![] }
        fn regions_in_layer(&self, _: Key<RegionLayer>) -> usize {
            panic!("No region layers exist!")
        }
        fn cells_in_region(&self, _: Key<RegionLayer>, _: usize) -> usize {
            panic!("No region layers exist!")
        }
        fn region_iter(&self, _: Key<RegionLayer>, _: usize) -> Self::Iter<'_> {
            panic!("No region layers exist!")
        }
        fn enclosing_region_and_offset(&self, _: Key<RegionLayer>, _: Index) -> Option<(usize, usize)> {
            panic!("No region layers exist!")
        }
        fn nth_in_region(&self, _: Key<RegionLayer>, _: usize, _: usize) -> Option<Index> {
            panic!("No region layers exist!")
        }
        fn mutually_visible(&self, _: Index, _: Index) -> bool { true }
        fn parse_state<V: Value>(&self, _: &str) -> Result<State<V, Self>, Error> {
            panic!("Not implemented")
        }
        fn serialize_state<V: Value>(&self, s: &State<V, Self>) -> String {
            (0..N).map(|c| {
                if let Some(v) = s.get([0, c]) {
                    v.to_string()
                } else {
                    ".".to_string()
                }
            }).collect::<Vec<_>>().join("")
        }
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
    pub fn replay_givens<V: Value, O: Overlay>(state: &mut State<V, O>) {
        for (i, v) in state.given_actions() {
            state.apply(i, v).unwrap();
        }
    }
}