#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --array=1-40
#SBATCH --partition=mcdermott

start=`date +%s`

cd /om2/user/mcusi/gen-bayesian-auditory-scenes/
source activate gentle
chapter=$((${SLURM_ARRAY_TASK_ID}-1))
dataset='gen4'
export JULIA_PROJECT=/om2/user/mcusi/gen-bayesian-auditory-scenes/
../julia-1.1.1/bin/julia ./dream-code/cleanup.jl $dataset $chapter

end=`date +%s`
echo "Duration: $((($(date +%s)-$start)/60)) minutes"

