set -euo pipefail
source ./scripts/lib.sh

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
ID_BASE=$(snapshot_stats --exist_skip)
git checkout "$EXP"
ID_EXP=$(snapshot_stats --exist_skip)
git checkout - >/dev/null 2>&1
mkdir -p "stats/${ID_EXP}-commit-diff"
for PUZZLE in $(list_puzzles); do
  touch "stats/${ID_BASE}/${PUZZLE}.json"
  touch "stats/${ID_EXP}/${PUZZLE}.json"
  cargo run --release --bin diff-stat -- \
    "stats/${ID_BASE}/${PUZZLE}.json" "stats/${ID_EXP}/${PUZZLE}.json" \
    > "stats/${ID_EXP}-commit-diff/${PUZZLE}.json"
done
for BENCH in $(list_benchmarks); do
  touch "stats/${ID_BASE}/${BENCH}.json"
  touch "stats/${ID_EXP}/${BENCH}.json"
  cargo run --release --bin diff-bench -- \
    "stats/${ID_BASE}/${BENCH}.json" "stats/${ID_EXP}/${BENCH}.json" \
    > "stats/${ID_EXP}-commit-diff/${BENCH}.json"
done
diff "stats/${ID_BASE}/summary.txt" "stats/${ID_EXP}/summary.txt" \
  > "stats/${ID_EXP}-commit-diff/summary.txt"
cat "stats/${ID_EXP}-commit-diff/summary.txt"
if [[ -t 0 ]]; then
  echo "Press any key to exit..."
  read -n 1 -s
fi
