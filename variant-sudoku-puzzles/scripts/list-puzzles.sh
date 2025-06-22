for PUZZLE in variant-sudoku-puzzles/src/bin/*.rs; do
  PUZZLE=$(basename "$PUZZLE" .rs)
  if [[ "$PUZZLE" == "stat-diff" ]]; then
    continue
  fi
  echo "$PUZZLE"
done