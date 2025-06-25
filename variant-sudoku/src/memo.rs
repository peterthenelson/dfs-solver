use std::collections::{HashMap, hash_map::Entry};
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::MutexGuard;

pub trait MemoCalc<A, K: Eq + Clone + Hash + 'static, V: 'static> {
    fn key(&self, args: &A) -> K;
    fn val(&self, args: &A) -> V;
}

pub struct FnToCalc<A, K: Eq + Clone + Hash + 'static, V: 'static>(fn (&A) -> K, fn (&A) -> V);
impl <A, K: Eq + Clone + Hash + 'static, V: 'static> FnToCalc<A, K, V> {
    pub fn new(kf: fn (&A) -> K, vf: fn (&A) -> V) -> Self {
        Self(kf, vf)
    }
}
impl <A, K: Eq + Clone + Hash + 'static, V: 'static> MemoCalc<A, K, V> for FnToCalc<A, K, V> {
    fn key(&self, args: &A) -> K { (self.0)(args) }
    fn val(&self, args: &A) -> V { (self.1)(args) }
}

pub struct MemoLock<A, K: Eq + Clone + Hash + 'static, V: 'static, C: MemoCalc<A, K, V>> {
    guard: MutexGuard<'static, HashMap<K, V>>,
    calc: C,
    _marker: PhantomData<A>,
}
impl <A, K: Eq + Clone + Hash + 'static, V: 'static, C: MemoCalc<A, K, V>> MemoLock<A, K, V, C> {
    pub fn new(guard: MutexGuard<'static, HashMap<K, V>>, calc: C) -> Self {
        Self { guard, calc, _marker: PhantomData }
    }

    pub fn get(&mut self, args: &A) -> &V {
        let calc = &self.calc;
        let k = calc.key(args);
        match self.guard.entry(k.clone()) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(v) => {
                let y = calc.val(args);
                v.insert(y)
            }
        }
    }
}
