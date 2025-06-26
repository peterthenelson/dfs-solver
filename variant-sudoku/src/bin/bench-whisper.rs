use rand::Rng;
use variant_sudoku::{bench::Bench, core::{VBitSet, VSet, Value}, sudoku::StdVal, whispers::{whisper_between, whisper_neighbors, whisper_possible_values}};
use rand_chacha::ChaCha20Rng;

// Trivially do something with a VBitSet just so nothing gets optimized away.
fn touch_set<const MIN: u8, const MAX: u8, VS: VSet<StdVal<MIN, MAX>>>(s: &VS) {
    for v in s.iter() {
        assert!(MIN <= v.val());
        assert!(v.val() <= MAX);
    }
}

struct Whisper<const MIN: u8, const MAX: u8, const DIST: u8>;

impl <const MIN: u8, const MAX: u8, const DIST: u8> Whisper<MIN, MAX, DIST> {
    fn sample_val(&self, rng: &mut ChaCha20Rng) -> StdVal<MIN, MAX> {
        StdVal::<MIN, MAX>::new(rng.random_range(MIN..=MAX))
    }
}

trait WhisperBench {
    fn run_whisper_neighbors(&self, rng: &mut ChaCha20Rng);
    fn run_whisper_possible_values(&self, rng: &mut ChaCha20Rng);
    fn run_whisper_between(&self, rng: &mut ChaCha20Rng);
}

impl <const MIN: u8, const MAX: u8, const DIST: u8>
WhisperBench for Whisper<MIN, MAX, DIST> {
    fn run_whisper_neighbors(&self, rng: &mut ChaCha20Rng) {
        let val = self.sample_val(rng);
        touch_set::<MIN, MAX, _>(&whisper_neighbors::<MIN, MAX>().get(DIST, val));
    }

    fn run_whisper_possible_values(&self, rng: &mut ChaCha20Rng) {
        let h2mvn = rng.random_bool(0.5);
        touch_set::<MIN, MAX, _>(&whisper_possible_values::<MIN, MAX>().get(DIST, h2mvn));
    }

    fn run_whisper_between(&self, rng: &mut ChaCha20Rng) {
        let set_sizes = 1..StdVal::<MIN, MAX>::cardinality();
        let mut left_set = VBitSet::<StdVal<MIN, MAX>>::empty();
        for _ in 0..rng.random_range(set_sizes.clone()) {
            left_set.insert(&self.sample_val(rng));
        }
        let mut right_set = VBitSet::<StdVal<MIN, MAX>>::empty();
        for _ in 0..rng.random_range(set_sizes) {
            right_set.insert(&self.sample_val(rng));
        }
        let result = whisper_between::<MIN, MAX, _>(DIST, &left_set, &right_set);
        touch_set::<MIN, MAX, _>(&result);
    }
}

fn main() {
    let mut bench = Bench::new();
    let ranges: Vec<Box<dyn WhisperBench>> = vec_box::vec_box![
        Whisper::<1, 4, 2>,
        Whisper::<1, 6, 3>,
        Whisper::<1, 8, 4>,
        Whisper::<1, 9, 5>,
        Whisper::<3, 9, 2>,
        Whisper::<10, 20, 5>,
        Whisper::<6, 12, 2>,
    ];
    bench.benchmark_cases(1000000, &ranges, "whisper_neighbors", |range, rng| {
        range.run_whisper_neighbors(rng);
    });
    bench.benchmark_cases(1000000, &ranges, "whisper_possible_values", |range, rng| {
        range.run_whisper_possible_values(rng);
    });
    bench.benchmark_cases(100000, &ranges, "whisper_between", |range, rng| {
        range.run_whisper_between(rng);
    });
    bench.save_json("stats/bench-whisper.json").unwrap()
}