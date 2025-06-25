use std::{fs::File, io::Read};
use variant_sudoku::debug::StatsSummary;

// Treats empty files as empty stats, but treats non-existent files as Errors.
fn read_json_file_or_empty_stats(path: &str) -> std::io::Result<StatsSummary> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let stats = if contents.trim().is_empty() {
        StatsSummary::default()
    } else {
        serde_json::from_str(&contents).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData, 
                format!("JSON error: {}", e),
            )
        })?
    };
    Ok(stats)
}

pub fn main() -> Result<(), std::io::Error> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Usage: diff-stat left.json right.json",
        ));
    }
    let left_stats = read_json_file_or_empty_stats(&args[1])?;
    let right_stats = read_json_file_or_empty_stats(&args[2])?;
    let diff = right_stats.delta_from(&left_stats);
    println!("{}", serde_json::to_string_pretty(&diff)?);
    Ok(())
}