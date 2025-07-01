set -euo pipefail
source ./scripts/lib.sh
for PUZZLE in $(list_puzzles); do
  cargo run --release --bin "$PUZZLE" -- --interactive
done
