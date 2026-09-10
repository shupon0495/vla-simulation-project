#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT=${PROJECT:-$SCRIPT_DIR}
source "$PROJECT/slurm/config.sh"

mkdir -p "$LOG"

RUN_ID=$(date +%s%N | sha256sum | cut -c1-10)
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$PROJECT/data/outputs/${TIMESTAMP}-${RUN_ID}"
mkdir -p "$RUN_DIR/manifests"
export RUN_ID RUN_DIR
printf '{\n  "run_id": "%s",\n  "timestamp": "%s",\n  "run_dir": "%s",\n  "preprocess_job_id": null,\n  "train_job_id": null,\n  "test_job_id": null\n}\n' "$RUN_ID" "$TIMESTAMP" "$RUN_DIR" > "$RUN_DIR/manifests/run.json"

# ログインノードでアセット準備を実行する前に、必要な Singularity イメージを作成する。
# アセット準備には Internet 接続が必要なため、build ジョブの完了を待機し、
# preprocess/train/test ジョブを投入する前に必ず完了させる。
BUILD_JOB=""
if [ ! -f "$SIF" ] || [ "$DEF" -nt "$SIF" ]; then
    echo "Building Singularity image..."
    BUILD_JOB=$(sbatch --wait --parsable \
        --export="ALL,PROJECT=$PROJECT,LOG=$LOG,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR" \
        --partition="$PPC_PARTITION" \
        --output="$LOG/build-%j-%Y-%m-%d.out" \
        --error="$LOG/build-%j-%Y-%m-%d.err" \
        "$PROJECT/slurm/build.sh"
    )
else
    echo "Singularity image is up to date."
fi

echo "Preparing offline assets..."
singularity exec --bind "$PROJECT:$PROJECT" --pwd "$PROJECT" "$SIF" \
    uv run python -m vla_simulation_project.main prepare-assets

JOB1=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR" \
    --partition="$PPC_PARTITION" \
    --output="$LOG/preprocess-%j-%Y-%m-%d.out" \
    --error="$LOG/preprocess-%j-%Y-%m-%d.err" \
    "$PROJECT/slurm/preprocess.sh"
)

JOB2=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR" \
    --partition="$TRAIN_PARTITION" \
    --output="$LOG/train-%j-%Y-%m-%d.out" \
    --error="$LOG/train-%j-%Y-%m-%d.err" \
    --dependency="afterok:$JOB1" \
    "$PROJECT/slurm/train.sh"
)

JOB3=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR" \
    --partition="$TEST_PARTITION" \
    --output="$LOG/test-%j-%Y-%m-%d.out" \
    --error="$LOG/test-%j-%Y-%m-%d.err" \
    --dependency="afterok:$JOB2" \
    "$PROJECT/slurm/test.sh"
)

printf '{\n  "run_id": "%s",\n  "timestamp": "%s",\n  "run_dir": "%s",\n  "preprocess_job_id": "%s",\n  "train_job_id": "%s",\n  "test_job_id": "%s"\n}\n' "$RUN_ID" "$TIMESTAMP" "$RUN_DIR" "$JOB1" "$JOB2" "$JOB3" > "$RUN_DIR/manifests/run.json"

echo "build:      ${BUILD_JOB:-skipped}"
echo "preprocess: $JOB1"
echo "train:      $JOB2"
echo "test:       $JOB3"
