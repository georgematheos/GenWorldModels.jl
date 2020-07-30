#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=40G
#SBATCH --partition=mcdermott

start=`date +%s`

cd /om2/user/mcusi/gen-bayesian-auditory-scenes/
source activate gentle
chapter=1 #$((${SLURM_ARRAY_TASK_ID}-1))
export JULIA_PROJECT=/om2/user/mcusi/gen-bayesian-auditory-scenes/
../julia-1.1.1/bin/julia dream.jl 'vec' 'vec' $chapter 20

end=`date +%s`
echo "Duration: $((($(date +%s)-$start)/60)) minutes"

###not needed SBATCH --array=1-40
