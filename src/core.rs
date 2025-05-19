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

pub trait UInt: PrimInt + Unsigned + TryInto<usize> {
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
pub struct UValWrapped;
pub struct UValUnwrapped;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UVal<U: UInt, S> {
    u: U,
    _state: PhantomData<S>,
}

impl <U: UInt> UVal<U, UValWrapped> {
    pub fn new(v: U) -> Self {
        UVal { u: v, _state: PhantomData }
    }
}

impl <U: UInt> UVal<U, UValUnwrapped> {
    pub fn value(&self) -> U {
        self.u
    }
}

impl <U: UInt> UVal<U, UValWrapped> {
    pub(self) fn unwrap(self) -> UVal<U, UValUnwrapped> {
        UVal { u: self.u, _state: PhantomData }
    }
}

/// Values in puzzles are drawn from a finite set of possible values. They are
/// represented as unsigned integers, but it's entirely up to the Value, State,
/// and Constraint implementations to interpret them.
pub trait Value<U: UInt>: Copy + Clone + Debug + PartialEq + Eq {
    fn cardinality() -> usize;
    fn parse(s: &str) -> Result<Self, Error>;

    fn from_uval(u: UVal<U, UValUnwrapped>) -> Self;
    fn to_uval(self) -> UVal<U, UValWrapped>;
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

    pub fn get(&self, index: Index) -> Option<UVal<U, UValWrapped>> {
        self.grid[index[0] * self.cols + index[1]].map(|v| UVal::new(v))
    }

    pub fn set(&mut self, index: Index, value: Option<UVal<U, UValWrapped>>) {
        self.grid[index[0] * self.cols + index[1]] = value.map(|v| v.unwrap().value());
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// This a set of values (e.g., that are possible, that have been seen, etc.).
/// They are represented as a bitset of the possible values.
#[derive(Debug, Clone)]
pub struct Set<U: UInt> {
    s: BitSet,
    _p_u: PhantomData<U>,
}

pub fn empty_set<U: UInt, V: Value<U>>() -> Set<U> {
    Set {
        s: BitSet::with_capacity(V::cardinality()),
        _p_u: PhantomData,
    }
}

pub fn full_set<U: UInt, V: Value<U>>() -> Set<U> {
    assert!(V::cardinality() <= 64, "Cardinality must be <= 64 for full_set");
    let mut s = Set {
        s: BitSet::with_capacity(V::cardinality()),
        _p_u: PhantomData,
    };
    let mask: u64 = (((1 as u128) << V::cardinality()) - 1) as u64;
    s.s.union_with(&BitSet::from_bytes(&mask.to_ne_bytes()));
    s
}

impl <U: UInt> Set<U> {
    pub fn insert(&mut self, value: UVal<U, UValWrapped>) {
        self.s.insert(value.unwrap().value().as_usize());
    }

    pub fn remove(&mut self, value: UVal<U, UValWrapped>) {
        self.s.remove(value.unwrap().value().as_usize());
    }

    pub fn contains(&self, value: UVal<U, UValWrapped>) -> bool {
        self.s.contains(value.unwrap().value().as_usize())
    }

    pub fn is_empty(&self) -> bool {
        self.s.is_empty()
    }

    pub fn clear(&mut self) {
        self.s.clear();
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = UVal<U, UValWrapped>> + 'a {
        self.s.iter().map(|i| UVal::new(U::from_usize(i)))
    }

    pub fn intersect_with(&mut self, other: &Set<U>) {
        self.s.intersect_with(&other.s);
    }

    pub fn union_with(&mut self, other: &Set<U>) {
        self.s.union_with(&other.s);
    }
}

/// This is a grid of Sets of values. It is used to represent the
/// not-yet-ruled-out values for each cell in the grid.
#[derive(Debug, Clone)]
pub struct SetGrid<U: UInt, V: Value<U>> {
    rows: usize,
    cols: usize,
    grid: Box<[Set<U>]>,
    _p_v: PhantomData<V>,
}

impl<U: UInt, V: Value<U>> SetGrid<U, V> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            grid: vec![empty_set::<U, V>(); rows * cols].into_boxed_slice(),
            _p_v: PhantomData,
        }
    }

    pub fn get(&self, index: Index) -> &Set<U> {
        &self.grid[index[0] * self.cols + index[1]]
    }

    pub fn get_mut(&mut self, index: Index) -> &mut Set<U> {
        self.grid.get_mut(index[0] * self.cols + index[1]).unwrap()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// This converts an extracted item from a container a Value, making use of the
/// private API to do so.
pub fn to_value<U: UInt, V: Value<U>>(u: UVal<U, UValWrapped>) -> V {
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
    fn undo(&mut self, index: Index) -> Result<(), Error> {
        let _ = index;
        Ok(())
    }
}

/// Trait for representing whatever puzzle is being solved in its current state
/// of being (partially) filled in. Ultimately this is just wrapping a Grid, but
/// it may impose additional meanings on the values of the grid.
pub trait State<U: UInt>: Clone + Debug {
    type Value: Value<U>;
    const ROWS: usize;
    const COLS: usize;
    fn reset(&mut self);
    fn get(&self, index: Index) -> Option<Self::Value>;
    fn apply(&mut self, index: Index, value: Self::Value) -> Result<(), Error>;
    fn undo(&mut self, index: Index) -> Result<(), Error>;
}

impl <U: UInt, S: State<U>> Stateful<U, S::Value> for S {}