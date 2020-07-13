#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --partition=mcdermott

start=`date +%s`

cd /om2/user/mcusi/gen-bayesian-auditory-scenes/
source activate gentle
export JULIA_PROJECT=/om2/user/mcusi/gen-bayesian-auditory-scenes/

dataset=$1
chapter=$((${SLURM_ARRAY_TASK_ID}-1))
../julia-1.1.1/bin/julia ./dream/cleanup.jl $dataset $chapter

end=`date +%s`
echo "Duration: $((($(date +%s)-$start)/60)) minutes"

