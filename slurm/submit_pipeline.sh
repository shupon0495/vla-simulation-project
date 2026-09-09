#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
mkdir -p "$LOG" "$DATA_DIR"
command -v sbatch >/dev/null || { echo "sbatch was not found" >&2; exit 1; }
command -v singularity >/dev/null || { echo "singularity was not found" >&2; exit 1; }
if [[ ! -f "$SIF" || "$DEF" -nt "$SIF" ]]; then
  echo "Building Singularity image with fakeroot: $SIF"
  singularity build --fakeroot --force "$SIF" "$DEF"
fi
RUN_STAGE="$PROJECT/slurm/run_stage.sh"
export PROJECT LOG DATA_DIR CONTAINER UV_CACHE_DIR RUN_STAGE
COMMON=(--export=ALL,PROJECT="$PROJECT",LOG="$LOG",DATA_DIR="$DATA_DIR",CONTAINER="$CONTAINER",UV_CACHE_DIR="$UV_CACHE_DIR",RUN_STAGE="$RUN_STAGE" --partition="$PARTITION" --time="$TIME_LIMIT" --cpus-per-task="$CPUS_PER_TASK" --mem="$MEMORY" --output="$LOG/%x-%j.out" --error="$LOG/%x-%j.err")
preprocess_id=$(sbatch --parsable "${COMMON[@]}" "$SCRIPT_DIR/preprocess.sh")
train_id=$(sbatch --parsable "${COMMON[@]}" --dependency="afterok:$preprocess_id" "$SCRIPT_DIR/train.sh")
test_id=$(sbatch --parsable "${COMMON[@]}" --dependency="afterok:$train_id" "$SCRIPT_DIR/test.sh")
printf 'preprocess: %s\ntrain:      %s\ntest:       %s\n' "$preprocess_id" "$train_id" "$test_id"
