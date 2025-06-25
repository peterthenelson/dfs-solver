for BENCH in variant-sudoku/src/bin/*.rs; do
  BENCH=$(basename "$BENCH" .rs)
  if [[ "$BENCH" == "diff-bench" ]]; then
    continue
  elif [[ "$BENCH" == "diff-stat" ]]; then
    continue
  fi
  echo "$BENCH"
done