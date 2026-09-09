#!/bin/bash

JOB1=$(sbatch --parsable preprocess.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 train.sh)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 test.sh)

echo "preprocess: $JOB1"
echo "train:      $JOB2"
echo "test:       $JOB3"