#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT=${PROJECT:-$SCRIPT_DIR}
source "$PROJECT/slurm/config.sh"

mkdir -p "$LOG"

# sifがないかdefのほうがsifより新しいときにdefを作成する
# もしloginノード内でbuildをするのが禁止されていたらjobに変更するようにする
BUILD_JOB=""
BUILD_DEPENDENCY=()
if [ ! -f "$SIF" ] || [ "$DEF" -nt "$SIF" ]; then
    echo "Building Singularity image..."
    BUILD_JOB=$(sbatch --parsable \
        --export="ALL,PROJECT=$PROJECT,LOG=$LOG" \
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
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG" \
    --partition="$PPC_PARTITION" \
    --output="$LOG/preprocess-%j-%Y-%m-%d.out" \
    --error="$LOG/preprocess-%j-%Y-%m-%d.err" \
    "${BUILD_DEPENDENCY[@]}" \
    "$PROJECT/slurm/preprocess.sh"
)

JOB2=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG" \
    --partition="$TRAIN_PARTITION" \
    --output="$LOG/train-%j-%Y-%m-%d.out" \
    --error="$LOG/train-%j-%Y-%m-%d.err" \
    --dependency="afterok:$JOB1" \
    "$PROJECT/slurm/train.sh"
)

JOB3=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG" \
    --partition="$TEST_PARTITION" \
    --output="$LOG/test-%j-%Y-%m-%d.out" \
    --error="$LOG/test-%j-%Y-%m-%d.err" \
    --dependency="afterok:$JOB2" \
    "$PROJECT/slurm/test.sh"
)

echo "build:      ${BUILD_JOB:-skipped}"
echo "preprocess: $JOB1"
echo "train:      $JOB2"
echo "test:       $JOB3"
