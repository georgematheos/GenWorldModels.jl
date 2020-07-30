#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=mcdermott

start=`date +%s`

cd /om2/user/mcusi/gen-bayesian-auditory-scenes/
source activate gentle
export JULIA_PROJECT=/om2/user/mcusi/gen-bayesian-auditory-scenes/
../julia-1.1.1/bin/julia ./dream-code/hear.jl 'gen4'

end=`date +%s`
echo "Duration: $((($(date +%s)-$start)/60)) minutes"
