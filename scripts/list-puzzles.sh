for PUZZLE in variant-sudoku-puzzles/src/bin/*.rs; do
  PUZZLE=$(basename "$PUZZLE" .rs)
  echo "$PUZZLE"
done