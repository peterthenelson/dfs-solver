use std::collections::{HashMap, hash_map::Entry};
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::MutexGuard;

pub trait MemoCalc<A, K: Eq + Clone + Hash + 'static, V: 'static> {
    fn key(&self, args: &A) -> K;
    fn calc(&self, key: &K) -> V;
}

pub struct MemoLock<A, K: Eq + Clone + Hash + 'static, V: 'static, C: MemoCalc<A, K, V>> {
    guard: MutexGuard<'static, HashMap<K, V>>,
    calc: C,
    _marker: PhantomData<A>,
}
impl <A, K: Eq + Clone + Hash + 'static, V: 'static, C: MemoCalc<A, K, V>> MemoLock<A, K, V, C> {
    pub fn get(&mut self, args: &A) -> &V {
        let calc = &self.calc;
        let k = calc.key(args);
        match self.guard.entry(k.clone()) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(v) => {
                let y = calc.calc(&k);
                v.insert(y)
            }
        }
    }
}

/* EXAMPLE:
static FOO: LazyLock<Mutex<HashMap<(u8, u8, u8, bool), String>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});
struct StrRepCalc<const MIN: u8, const MAX: u8>;
impl <const MIN: u8, const MAX: u8> MemoCalc<(u8, bool), (u8, u8, u8, bool), String> for StrRepCalc<MIN, MAX> {
    fn key(&self, args: &(u8, bool)) -> (u8, u8, u8, bool) { (MIN, MAX, args.0, args.1) }
    fn calc(&self, key: &(u8, u8, u8, bool)) -> String {
        format!("key={:?}", key)
    }
}
fn foo<const MIN: u8, const MAX: u8>() -> MemoLock<(u8, bool), (u8, u8, u8, bool), String, StrRepCalc<MIN, MAX>> {
    MemoLock {
        guard: FOO.try_lock().expect("Must release lock before calling foo again"),
        calc: StrRepCalc::<MIN, MAX>,
        _marker: PhantomData,
    }
}

pub fn memo_main() {
    {
        let mut f = foo::<1, 9>();
        println!("{}", f.get(&(1, false)));
        println!("{}", f.get(&(2, true)));
    }
    {
        let mut f = foo::<1, 4>();
        println!("{}", f.get(&(1, false)));
        println!("{}", f.get(&(2, true)));
    }
    {
        let mut f = foo::<1, 9>();
        println!("{}", f.get(&(1, false)));
        println!("{}", f.get(&(2, true)));
    }
}
*/
