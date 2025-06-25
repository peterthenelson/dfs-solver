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
if [[ -d "stats/$HASH" ]]; then
  case "$EXIST" in
  skip)
    echo "Stats for $HASH already exists; skipping." >&2
    exit 0
    ;;
  redo)
    echo "Stats for $HASH already exists; redoing." >&2
    rm stats/$HASH/*
    ;;
  *)
    echo "Stats for $HASH already exists!" >&2
    exit 1
    ;;
  esac
else
  mkdir "stats/$HASH"
fi
echo "Stats for commit $HASH" >> "stats/$HASH/summary.txt"
echo "------------------------" >> "stats/$HASH/summary.txt"
cargo build --release
for PUZZLE in $(./scripts/list-puzzles.sh); do
  cargo run --release --bin "$PUZZLE" 1>> "stats/$HASH/summary.txt"
  mv "stats/$PUZZLE.png" "stats/$HASH"
  mv "stats/$PUZZLE.json" "stats/$HASH"
done
for BENCH in $(./scripts/list-benches.sh); do
  cargo run --release --bin "$BENCH" 1>> "stats/$HASH/summary.txt"
  mv "stats/$BENCH.json" "stats/$HASH"
done
