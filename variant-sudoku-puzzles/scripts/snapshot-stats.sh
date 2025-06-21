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
hash=$(git describe --always --dirty)
echo "$hash"
if [[ -d "figures/$hash" ]]; then
  case "$EXIST" in
  skip)
    echo "Stats for $hash already exists; skipping." >&2
    exit 0
    ;;
  redo)
    echo "Stats for $hash already exists; redoing." >&2
    rm figures/$hash/*
    ;;
  *)
    echo "Stats for $hash already exists!" >&2
    exit 1
    ;;
  esac
else
  mkdir "figures/$hash"
fi
echo "Stats for commit $hash" >> "figures/$hash/summary.txt"
echo "------------------------" >> "figures/$hash/summary.txt"
cargo build --release
for puzzle in $(./variant-sudoku-puzzles/scripts/list-puzzles.sh); do
  cargo run --release --bin "$puzzle" 1>> "figures/$hash/summary.txt"
  mv "figures/$puzzle.png" "figures/$hash"
  mv "figures/$puzzle.json" "figures/$hash"
done
