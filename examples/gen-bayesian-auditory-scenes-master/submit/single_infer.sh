#!/bin/bash
#SBATCH --time=02:00:00

fn='1A'
echo "name of observation: $fn"

cd /om2/user/mcusi/gen-bayesian-auditory-scenes/
source activate gentle
export JULIA_PROJECT=/om2/user/mcusi/gen-bayesian-auditory-scenes/
../julia-1.1.1/bin/julia ./inference/infer.jl $fn guide