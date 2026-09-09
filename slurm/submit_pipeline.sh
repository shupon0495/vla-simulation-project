#!/bin/bash
set -e # エラーで終了するように設定

export PROJECT=/home/users/$USER/vla-simulation-project
source $PROJECT/slurm/config.sh

# sifがないかdefのほうがsifより新しいときにdefを作成する
# もしloginノード内でbuildをするのが禁止されていたらjobに変更するようにする
if [ ! -f "$SIF" ] || [ "$DEF" -nt "$SIF" ]; then
    echo "Building Singularity image..."
    $(sbatch --parsable --export=ALL,PROJECT=$PROJECT,LOG=$LOG $PROJECT/slurm/build.sh)
else
    echo "Singularity image is up to date."
fi
JOB1=$(sbatch --parsable --export=ALL,PROJECT=$PROJECT,LOG=$LOG --partition=$PPC_PARTITION $PROJECT/slurm/preprocess.sh)
JOB2=$(sbatch --parsable --export=ALL,PROJECT=$PROJECT, LOG=$LOG --partition=$TRAIN_PARTITION --dependency=afterok:$JOB1 $PROJECT/slurm/train.sh)
JOB3=$(sbatch --parsable --export=ALL,PROJECT=$PROJECT, LOG=$LOG --partition=$TEST_PARTITION --dependency=afterok:$JOB2 $PROJECT/slurm/test.sh)

echo "preprocess: $JOB1"
echo "train:      $JOB2"
echo "test:       $JOB3"