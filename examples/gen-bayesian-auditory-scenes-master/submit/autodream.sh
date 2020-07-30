#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --partition=mcdermott

start=`date +%s`

cd /om2/user/mcusi/gen-bayesian-auditory-scenes/
source activate gentle
export JULIA_PROJECT=/om2/user/mcusi/gen-bayesian-auditory-scenes/

dataset=$1
paramfile=$2
chapter=$((${SLURM_ARRAY_TASK_ID}-1))
n=$3
../julia-1.1.1/bin/julia ./dream-code/dream.jl $dataset $paramfile $chapter $n

end=`date +%s`
echo "Duration: $((($(date +%s)-$start)/60)) minutes"
