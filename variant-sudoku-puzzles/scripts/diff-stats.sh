set -euo pipefail
git add .
if git diff --cached --quiet; then
  echo "No staged changes to compare against base."
  exit 1
fi
git stash push -m "Temporary stash for diff-stats"
ID_BASE=$(./variant-sudoku-puzzles/scripts/snapshot-stats.sh --exist_skip)
git stash pop
ID_EXP=$(./variant-sudoku-puzzles/scripts/snapshot-stats.sh --exist_redo)
mkdir -p "figures/${ID_BASE}-diff"
for puzzle in $(./variant-sudoku-puzzles/scripts/list-puzzles.sh); do
  cargo run --release --bin stat-diff -- \
    "figures/${ID_BASE}/${puzzle}.json" "figures/${ID_EXP}/${puzzle}.json" \
    > "figures/${ID_BASE}-diff/${puzzle}.json"
done
diff "figures/${ID_BASE}/summary.txt" "figures/${ID_EXP}/summary.txt" \
  > "figures/${ID_BASE}-diff/summary.txt"
echo "figures/${ID_BASE}-diff/summary.txt" | grep "Steps.*"
if [[ -t 0 ]]; then
  echo "Press any key to exit..."
  read -n 1 -s
fi
