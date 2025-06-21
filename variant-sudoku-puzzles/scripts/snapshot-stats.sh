set -euo pipefail
git add .
hash=$(git describe --always --dirty)
mkdir "figures/$hash"
echo "Stats for commit $hash" >> "figures/$hash/summary.txt"
echo "------------------------" >> "figures/$hash/summary.txt"
cargo build --release
for puzzle in variant-sudoku-puzzles/src/bin/*.rs; do
  puzzle=$(basename $puzzle .rs)
  cargo run --release --bin $puzzle 1>> "figures/$hash/summary.txt"
  mv "figures/$puzzle.png" "figures/$hash"
done