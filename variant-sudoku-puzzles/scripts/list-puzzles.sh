for puzzle in variant-sudoku-puzzles/src/bin/*.rs; do
  puzzle=$(basename "$puzzle" .rs)
  if [[ "$puzzle" == "stat-diff" ]]; then
    continue
  fi
  echo "$puzzle"
done