use std::{collections::BTreeMap, fs::File, io::Read};
use variant_sudoku::bench::diff_results;

pub fn main() -> Result<(), std::io::Error> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Usage: bench-diff left.json right.json",
        ));
    }
    let mut left_file = File::open(&args[1]).unwrap();
    let mut left_data = String::new();
    left_file.read_to_string(&mut left_data).unwrap();
    let left_results: BTreeMap<String, f64> = serde_json::from_str(&left_data)?;
    let mut right_file = File::open(&args[2]).unwrap();
    let mut right_data = String::new();
    right_file.read_to_string(&mut right_data).unwrap();
    let right_results: BTreeMap<String, f64> = serde_json::from_str(&right_data)?;
    let diff = diff_results(&left_results, &right_results);
    println!("{}", serde_json::to_string_pretty(&diff)?);
    Ok(())
}
