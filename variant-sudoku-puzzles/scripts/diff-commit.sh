set -euo pipefail
if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <commit-hash>"
  exit 1
fi

EXP="$1"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: You have uncommitted changes. Please commit or stash them before running this script."
  exit 1
fi

BRANCH=$(git symbolic-ref --short -q HEAD || echo "DETACHED")
if [[ "$BRANCH" == "DETACHED" ]]; then
  echo "Error: Currently in a detached HEAD state. Please return to some branch before running this script."
  exit 1
fi
cleanup() {
    git checkout "$BRANCH" >/dev/null 2>&1
}
trap cleanup EXIT

MSG=$(git log -1 --pretty=format:%s "$EXP")
echo "Running diff for commit: $MSG"
BASE=$(git rev-parse "${EXP}^")
git checkout "$BASE"
ID_BASE=$(./variant-sudoku-puzzles/scripts/snapshot-stats.sh --exist_skip)
git checkout "$EXP"
ID_EXP=$(./variant-sudoku-puzzles/scripts/snapshot-stats.sh --exist_skip)
git checkout - >/dev/null 2>&1
mkdir -p "figures/${ID_EXP}-commit-diff"
for PUZZLE in $(./variant-sudoku-puzzles/scripts/list-puzzles.sh); do
  cargo run --release --bin stat-diff -- \
    "figures/${ID_BASE}/${PUZZLE}.json" "figures/${ID_EXP}/${PUZZLE}.json" \
    > "figures/${ID_EXP}-commit-diff/${PUZZLE}.json"
done
diff "figures/${ID_BASE}/summary.txt" "figures/${ID_EXP}/summary.txt" \
  > "figures/${ID_EXP}-commit-diff/summary.txt"
cat "figures/${ID_EXP}-commit-diff/summary.txt" | grep "Steps:.*"
if [[ -t 0 ]]; then
  echo "Press any key to exit..."
  read -n 1 -s
fi
