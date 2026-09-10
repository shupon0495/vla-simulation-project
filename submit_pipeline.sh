#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT=${PROJECT:-$SCRIPT_DIR}
source "$PROJECT/slurm/config.sh"

mkdir -p "$LOG"

RUN_ID=$(date +%s%N | sha256sum | cut -c1-10)
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$PROJECT/data/outputs/${TIMESTAMP}-${RUN_ID}"
export RUN_ID TIMESTAMP RUN_DIR
mkdir -p "$RUN_DIR/manifests"

# The login node's system Python may be 3.9, while the project requires 3.12.
# Build and run a small, login-only image before submitting any Slurm job.  This
# image is intentionally separate from the compute image managed by build.sh.
if [ ! -f "$LOGIN_SIF" ] || [ "$LOGIN_DEF" -nt "$LOGIN_SIF" ]; then
    echo "Building login-node asset preparation image..."
    singularity build --fakeroot --force "$LOGIN_SIF" "$LOGIN_DEF"
else
    echo "Login-node asset preparation image is up to date."
fi

singularity exec \
    --bind "$PROJECT:$PROJECT" \
    --pwd "$PROJECT" \
    --env "PROJECT=$PROJECT,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR,PYTHONPATH=$PROJECT/src" \
    "$LOGIN_SIF" \
    "$PROJECT/.venv/bin/python" -m vla_simulation_project.main prepare-assets
printf '{\n  "run_id": "%s",\n  "timestamp": "%s",\n  "run_dir": "%s",\n  "build_job_id": null,\n  "preprocess_job_id": null,\n  "train_job_id": null,\n  "test_job_id": null\n}\n' \
    "$RUN_ID" "$TIMESTAMP" "$RUN_DIR" > "$RUN_DIR/manifests/run.json"

# sifがないかdefのほうがsifより新しいときにdefを作成する
# もしloginノード内でbuildをするのが禁止されていたらjobに変更するようにする
BUILD_JOB=""
BUILD_DEPENDENCY=()
if [ ! -f "$SIF" ] || [ "$DEF" -nt "$SIF" ]; then
    echo "Building Singularity image..."

    BUILD_JOB=$(sbatch --parsable \
        --export="ALL,PROJECT=$PROJECT,LOG=$LOG,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR" \
        --partition="$PPC_PARTITION" \
        --output="$LOG/build-%j-%Y-%m-%d.out" \
        --error="$LOG/build-%j-%Y-%m-%d.err" \
        "$PROJECT/slurm/build.sh"
    )
    BUILD_DEPENDENCY=(--dependency="afterok:$BUILD_JOB")
else
    echo "Singularity image is up to date."
fi

JOB1=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR,HF_HOME=$PROJECT/data/hf_cache,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1,TRANSFORMERS_OFFLINE=1" \
    --partition="$PPC_PARTITION" \
    --output="$LOG/preprocess-%j-%Y-%m-%d.out" \
    --error="$LOG/preprocess-%j-%Y-%m-%d.err" \
    "${BUILD_DEPENDENCY[@]}" \
    "$PROJECT/slurm/preprocess.sh"
)

JOB2=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR,HF_HOME=$PROJECT/data/hf_cache,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1,TRANSFORMERS_OFFLINE=1" \
    --partition="$TRAIN_PARTITION" \
    --output="$LOG/train-%j-%Y-%m-%d.out" \
    --error="$LOG/train-%j-%Y-%m-%d.err" \
    --dependency="afterok:$JOB1" \
    "$PROJECT/slurm/train.sh"
)

JOB3=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR,HF_HOME=$PROJECT/data/hf_cache,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1,TRANSFORMERS_OFFLINE=1" \
    --partition="$TEST_PARTITION" \
    --output="$LOG/test-%j-%Y-%m-%d.out" \
    --error="$LOG/test-%j-%Y-%m-%d.err" \
    --dependency="afterok:$JOB2" \
    "$PROJECT/slurm/test.sh"
)

BUILD_JSON=null
if [ -n "$BUILD_JOB" ]; then BUILD_JSON="\"$BUILD_JOB\""; fi
printf '{\n  "run_id": "%s",\n  "timestamp": "%s",\n  "run_dir": "%s",\n  "build_job_id": %s,\n  "preprocess_job_id": "%s",\n  "train_job_id": "%s",\n  "test_job_id": "%s"\n}\n' \
    "$RUN_ID" "$TIMESTAMP" "$RUN_DIR" "$BUILD_JSON" "$JOB1" "$JOB2" "$JOB3" \
    > "$RUN_DIR/manifests/run.json"

echo "build:      ${BUILD_JOB:-skipped}"
echo "preprocess: $JOB1"
echo "train:      $JOB2"
echo "test:       $JOB3"
