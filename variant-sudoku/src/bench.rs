use std::{collections::{BTreeMap, BTreeSet}, fs::File, io::Write, time::Instant};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub struct Bench {
    results: BTreeMap<String, f64>,
    rng: ChaCha20Rng,
}

const SEED: u64 = 0xeea42aa1638be961;

impl Bench {
    pub fn new() -> Self {
        Self {
            results: BTreeMap::new(),
            rng: ChaCha20Rng::seed_from_u64(SEED),
        }
    }

    pub fn benchmark<F: FnMut(&mut ChaCha20Rng) -> ()>(&mut self, n: usize, name: &str, mut f: F) {
        let start = Instant::now();
        for _ in 0..n {
            f(&mut self.rng);
        }
        let duration = start.elapsed();
        println!("{}: {:?}", name, duration);
        self.results.insert(name.into(), duration.as_secs_f64());

    }

    pub fn benchmark_cases<T, F: FnMut(&T, &mut ChaCha20Rng) -> ()>(&mut self, n: usize, cases: &Vec<T>, name: &str, mut f: F) {
        let start = Instant::now();
        for _ in 0..n {
            for case in cases {
                f(case, &mut self.rng);
            }
        }
        let duration = start.elapsed();
        println!("{}: {:?}", name, duration);
        self.results.insert(name.into(), duration.as_secs_f64());
    }

    pub fn into_results(self) -> BTreeMap<String, f64> {
        self.results
    }

    pub fn save_json(&self, filename: &str) -> Result<(), std::io::Error> {
        let mut f = File::create(filename)?;
        let json_data = serde_json::to_string_pretty(&self.results)?;
        f.write_all(json_data.as_bytes())?;
        Ok(())
    }
}

pub fn diff_results(
    left: &BTreeMap<String, f64>, right: &BTreeMap<String, f64>,
) -> BTreeMap<String, f64> {
    let all_keys: BTreeSet<_> = left.keys().chain(right.keys()).cloned().collect();
    all_keys
        .into_iter()
        .map(|k| {
            let l = left.get(&k).unwrap_or(&0.0);
            let r = right.get(&k).unwrap_or(&0.0);
            (k, r - l)
        })
        .collect()
}