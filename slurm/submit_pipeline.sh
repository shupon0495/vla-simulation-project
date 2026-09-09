#!/bin/bash
set -e # エラーで終了するように設定

export PROJECT=/home/users/$USER/vla-simulation-project
source $PROJECT/slurm/config.sh

# sifがないかdefのほうがsifより新しいときにdefを作成する
# もしloginノード内でbuildをするのが禁止されていたらjobに変更するようにする
if [ ! -f "$SIF" ] || [ "$DEF" -nt "$SIF" ]; then
    echo "Building Singularity image..."
    BUILD_JOB=$(sbatch --parsable \
        --export="ALL,PROJECT=$PROJECT,LOG=$LOG" \
        --partition="$PPC_PARTITION" \
        --output="$LOG/build-%j.out" \
        --error="$LOG/build-%j.err" \
        "$PROJECT/slurm/build.sh"
    )
else
    echo "Singularity image is up to date."
fi

JOB1=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG" \
    --partition="$PPC_PARTITION" \
    --output="$LOG/preprocess-%j.out" \
    --error="$LOG/preprocess-%j.err" \
    "$PROJECT/slurm/preprocess.sh"
)

JOB2=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG" \
    --partition="$TRAIN_PARTITION" \
    --output="$LOG/train-%j.out" \
    --error="$LOG/train-%j.err" \
    --dependency="afterok:$JOB1" \
    "$PROJECT/slurm/train.sh"
)

JOB3=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG" \
    --partition="$TEST_PARTITION" \
    --output="$LOG/test-%j.out" \
    --error="$LOG/test-%j.err" \
    --dependency="afterok:$JOB2" \
    "$PROJECT/slurm/test.sh"
)

echo "build:      $BUILD_JOB"
echo "preprocess: $JOB1"
echo "train:      $JOB2"
echo "test:       $JOB3"