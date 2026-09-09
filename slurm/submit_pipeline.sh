#!/bin/bash
set -e

PROJECT=/home/users/$USER/vla_simulation_source
DEF=$PROJECT/singularity/ubuntu24.04.def
SIF=$PROJECT/singularity/ubuntu24.04.sif
LOG=$PROJECT/log

# sifがないかdefのほうがsifより新しいときにdefを作成する
# もしloginノード内でbuildをするのが禁止されていたらjobに変更するようにする
if [ ! -f "$SIF" ] || [ "$DEF" -nt "$SIF" ]; then
    echo "Building Singularity image..."
    singularity build --force "$SIF" "$DEF"
else
    echo "Singularity image is up to date."
fi

JOB1=$(sbatch --parsable --export=ALL, PROJECT=$PROJECT, LOG=$LOG $PROJECT/slurm/preprocess.sh)
JOB2=$(sbatch --parsable --export=ALL, PROJECT=$PROJECT, LOG=$LOG --dependency=afterok:$JOB1 $PROJECT/slurm/train.sh)
JOB3=$(sbatch --parsable --export=ALL, PROJECT=$PROJECT, LOG=$LOG --dependency=afterok:$JOB2 $PROJECT/slurm/test.sh)

echo "preprocess: $JOB1"
echo "train:      $JOB2"
echo "test:       $JOB3"