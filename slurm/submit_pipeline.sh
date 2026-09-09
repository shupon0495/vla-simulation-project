#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
mkdir -p "$LOG" "$DATA_DIR"
command -v sbatch >/dev/null || { echo "sbatch was not found" >&2; exit 1; }
command -v singularity >/dev/null || { echo "singularity was not found" >&2; exit 1; }
if [[ "$CONTAINER" == "$SIF" ]]; then
  [[ -f "$DEF" ]] || { echo "Singularity definition was not found: $DEF" >&2; exit 1; }
  if [[ ! -f "$SIF" || "$DEF" -nt "$SIF" ]]; then
    echo "Building Singularity image with fakeroot: $SIF"
    singularity build --fakeroot --force "$SIF" "$DEF"
  fi
elif [[ ! -f "$CONTAINER" ]]; then
  echo "Container image was not found: $CONTAINER" >&2
  exit 1
fi
RUN_STAGE="$PROJECT/slurm/run_stage.sh"
export PROJECT LOG DATA_DIR CONTAINER UV_CACHE_DIR RUN_STAGE
COMMON=(--export=ALL,PROJECT="$PROJECT",LOG="$LOG",DATA_DIR="$DATA_DIR",CONTAINER="$CONTAINER",UV_CACHE_DIR="$UV_CACHE_DIR",RUN_STAGE="$RUN_STAGE" --partition="$PARTITION" --time="$TIME_LIMIT" --cpus-per-task="$CPUS_PER_TASK" --mem="$MEMORY" --output="$LOG/%x-%j.out" --error="$LOG/%x-%j.err")

submit_job() {
  local response
  response=$(sbatch --parsable "$@")
  # --parsable can return "job_id;cluster"; dependencies require the job ID only.
  printf '%s\n' "${response%%;*}"
}

preprocess_id=$(submit_job "${COMMON[@]}" "$SCRIPT_DIR/preprocess.sh")
train_id=$(submit_job "${COMMON[@]}" --dependency="afterok:$preprocess_id" "$SCRIPT_DIR/train.sh")
test_id=$(submit_job "${COMMON[@]}" --dependency="afterok:$train_id" "$SCRIPT_DIR/test.sh")
printf 'preprocess: %s\ntrain:      %s\ntest:       %s\n' "$preprocess_id" "$train_id" "$test_id"
