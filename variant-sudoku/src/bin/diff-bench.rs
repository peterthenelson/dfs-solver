use std::{collections::BTreeMap, fs::File, io::Read};
use variant_sudoku::bench::diff_results;

// Treats empty files as empty maps, but treats non-existent files as Errors.
fn read_json_file_or_empty_map(path: &str) -> std::io::Result<BTreeMap<String, f64>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let map = if contents.trim().is_empty() {
        BTreeMap::new()
    } else {
        serde_json::from_str(&contents).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData, 
                format!("JSON error: {}", e),
            )
        })?
    };
    Ok(map)
}

pub fn main() -> Result<(), std::io::Error> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Usage: diff-bench left.json right.json",
        ));
    }
    let left_results = read_json_file_or_empty_map(&args[1])?;
    let right_results = read_json_file_or_empty_map(&args[2])?;
    let diff = diff_results(&left_results, &right_results);
    println!("{}", serde_json::to_string_pretty(&diff)?);
    Ok(())
}
