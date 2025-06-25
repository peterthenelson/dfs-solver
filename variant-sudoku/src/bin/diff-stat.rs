use std::{fs::File, io::Read};
use variant_sudoku::debug::StatsSummary;

pub fn main() -> Result<(), std::io::Error> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Usage: stat-diff left.json right.json",
        ));
    }
    let mut left_file = File::open(&args[1]).unwrap();
    let mut left_data = String::new();
    left_file.read_to_string(&mut left_data).unwrap();
    let left_stats: StatsSummary = serde_json::from_str(&left_data)?;
    let mut right_file = File::open(&args[2]).unwrap();
    let mut right_data = String::new();
    right_file.read_to_string(&mut right_data).unwrap();
    let right_stats: StatsSummary = serde_json::from_str(&right_data)?;
    let diff = right_stats.delta_from(&left_stats);
    println!("{}", serde_json::to_string_pretty(&diff)?);
    Ok(())
}