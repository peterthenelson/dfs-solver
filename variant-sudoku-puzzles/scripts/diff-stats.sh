set -euo pipefail
git add .
if git diff --cached --quiet; then
  echo "No staged changes to compare against base."
  exit 1
fi
git stash push --keep-index -u -m "Temporary stash for diff-stats"
ID_BASE=$(./variant-sudoku-puzzles/scripts/snapshot-stats.sh)
git stash pop
ID_EXP=$(./variant-sudoku-puzzles/scripts/snapshot-stats.sh)
diff "figures/${ID_BASE}/summary.txt" "figures/${ID_EXP}/summary.txt"
