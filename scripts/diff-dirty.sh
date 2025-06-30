set -euo pipefail
source ./scripts/lib.sh
git add .
if git diff --cached --quiet; then
  echo "No staged changes to compare against base."
  exit 1
fi
git stash push -m "Temporary stash for diff-stats"
ID_BASE=$(snapshot_stats --exist_skip)
git stash pop
ID_EXP=$(snapshot_stats --exist_redo)
mkdir -p "stats/${ID_BASE}-diff"
for PUZZLE in $(list_puzzles); do
  touch "stats/${ID_BASE}/${PUZZLE}.json"
  touch "stats/${ID_EXP}/${PUZZLE}.json"
  cargo run --release --bin diff-stat -- \
    "stats/${ID_BASE}/${PUZZLE}.json" "stats/${ID_EXP}/${PUZZLE}.json" \
    > "stats/${ID_BASE}-diff/${PUZZLE}.json"
done
for BENCH in $(list_benchmarks); do
  touch "stats/${ID_BASE}/${BENCH}.json"
  touch "stats/${ID_EXP}/${BENCH}.json"
  cargo run --release --bin diff-bench -- \
    "stats/${ID_BASE}/${BENCH}.json" "stats/${ID_EXP}/${BENCH}.json" \
    > "stats/${ID_BASE}-diff/${BENCH}.json"
done
diff "stats/${ID_BASE}/summary.txt" "stats/${ID_EXP}/summary.txt" \
  > "stats/${ID_BASE}-diff/summary.txt"
cat "stats/${ID_BASE}-diff/summary.txt"
if [[ -t 0 ]]; then
  echo "Press any key to exit..."
  read -n 1 -s
fi
