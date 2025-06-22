set -euo pipefail

EXIST="fail"
if [[ $# -gt 0 ]]; then
  case "$1" in
    --exist_fail)
      EXIST="fail"
      ;;
    --exist_redo)
      EXIST="redo"
      ;;
    --exist_skip)
      EXIST="skip"
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--exist_fail|--exist_redo|--exist_skip]"
      exit 1
      ;;
  esac
fi

git add .
HASH=$(git describe --always --dirty)
echo "$HASH"
if [[ -d "figures/$HASH" ]]; then
  case "$EXIST" in
  skip)
    echo "Stats for $HASH already exists; skipping." >&2
    exit 0
    ;;
  redo)
    echo "Stats for $HASH already exists; redoing." >&2
    rm figures/$HASH/*
    ;;
  *)
    echo "Stats for $HASH already exists!" >&2
    exit 1
    ;;
  esac
else
  mkdir "figures/$HASH"
fi
echo "Stats for commit $HASH" >> "figures/$HASH/summary.txt"
echo "------------------------" >> "figures/$HASH/summary.txt"
cargo build --release
for PUZZLE in $(./variant-sudoku-puzzles/scripts/list-puzzles.sh); do
  cargo run --release --bin "$PUZZLE" 1>> "figures/$HASH/summary.txt"
  mv "figures/$PUZZLE.png" "figures/$HASH"
  mv "figures/$PUZZLE.json" "figures/$HASH"
done
