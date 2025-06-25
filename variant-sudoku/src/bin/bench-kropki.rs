use rand::Rng;
use variant_sudoku::{bench::Bench, core::{to_value, UVSet}, sudoku::StdVal, kropki::{kropki_black_possible, kropki_black_possible_chain}};
use rand_chacha::ChaCha20Rng;

// Trivially do something with a UVSet just so nothing gets optimized away.
fn touch_set<const MIN: u8, const MAX: u8>(s: &UVSet<u8>) {
    for uv in s.iter() {
        let v = to_value::<StdVal<MIN, MAX>>(uv);
        assert!(MIN <= v.val());
        assert!(v.val() <= MAX);
    }
}

struct Kropki<const MIN: u8, const MAX: u8>;

trait KropkiBench {
    fn run_kropki_black_possible(&self, rng: &mut ChaCha20Rng);
    fn run_kropki_black_possible_chain(&self, rng: &mut ChaCha20Rng);
}

impl <const MIN: u8, const MAX: u8>
KropkiBench for Kropki<MIN, MAX> {
    fn run_kropki_black_possible(&self, _: &mut ChaCha20Rng) {
        touch_set::<MIN, MAX>(&kropki_black_possible::<MIN, MAX>());
    }

    fn run_kropki_black_possible_chain(&self, rng: &mut ChaCha20Rng) {
        let n_mutually_visible = rng.random_range(2..(MAX-MIN)) as usize;
        let len_from_end = rng.random_range(0..n_mutually_visible) as usize;
        touch_set::<MIN, MAX>(&kropki_black_possible_chain::<MIN, MAX>(n_mutually_visible, len_from_end));
    }
}

fn main() {
    let mut bench = Bench::new();
    let ranges: Vec<Box<dyn KropkiBench>> = vec_box::vec_box![
        Kropki::<1, 4>,
        Kropki::<1, 6>,
        Kropki::<1, 8>,
        Kropki::<1, 9>,
        Kropki::<3, 9>,
        Kropki::<10, 20>,
        Kropki::<6, 12>,
    ];
    bench.benchmark_cases(1000000, &ranges, "kropki_black_possible", |range, rng| {
        range.run_kropki_black_possible(rng);
    });
    bench.benchmark_cases(1000000, &ranges, "kropki_black_possible_chain", |range, rng| {
        range.run_kropki_black_possible_chain(rng);
    });
    bench.save_json("stats/bench-kropki.json").unwrap()
}
