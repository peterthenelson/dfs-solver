set -euo pipefail

list_puzzles() {
  local puzzle
  for puzzle in variant-sudoku-puzzles/src/bin/*.rs; do
    puzzle=$(basename "$puzzle" .rs)
    echo "$puzzle"
  done
}

list_benchmarks() {
  local bench
  for bench in variant-sudoku/src/bin/*.rs; do
    bench=$(basename "$bench" .rs)
    if [[ "$bench" == "diff-bench" ]]; then
      continue
    elif [[ "$bench" == "diff-stat" ]]; then
      continue
    fi
    echo "$bench"
  done
}

snapshot_stats() {
  local exist hash puzzle bench
  exist="fail"
  if [[ $# -gt 0 ]]; then
    case "$1" in
      --exist_fail)
        exist="fail"
        ;;
      --exist_redo)
        exist="redo"
        ;;
      --exist_skip)
        exist="skip"
        ;;
      *)
        echo "Unknown option: $1"
        echo "Usage: $0 [--exist_fail|--exist_redo|--exist_skip]"
        exit 1
        ;;
    esac
  fi
  git add .
  hash=$(git describe --always --dirty)
  echo "$hash"
  if [[ -d "stats/$hash" ]]; then
    case "$exist" in
    skip)
      echo "Stats for $hash already exists; skipping." >&2
      exit 0
      ;;
    redo)
      echo "Stats for $hash already exists; redoing." >&2
      rm stats/$hash/*
      ;;
    *)
      echo "Stats for $hash already exists!" >&2
      exit 1
      ;;
    esac
  else
    mkdir "stats/$hash"
  fi
  echo "Stats for commit $hash" >> "stats/$hash/summary.txt"
  echo "------------------------" >> "stats/$hash/summary.txt"
  cargo build --release
  for puzzle in $(list_puzzles); do
    cargo run --release --bin "$puzzle" 1>> "stats/$hash/summary.txt"
    mv "stats/$puzzle.png" "stats/$hash"
    mv "stats/$puzzle.json" "stats/$hash"
  done
  for bench in $(list_benchmarks); do
    cargo run --release --bin "$bench" 1>> "stats/$hash/summary.txt"
    mv "stats/$bench.json" "stats/$hash"
  done
}

