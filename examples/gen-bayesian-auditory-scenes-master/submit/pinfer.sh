#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --partition=mcdermott
#SBATCH --array=0-12

sounds=("tougas_bregman_1A" "bregman_rudnicky" "bregman_rudnicky" "bregman_rudnicky" "bregman_rudnicky" "ABA" "ABA" "ABA" "ABA" "ABA" "ABA" "ABA" "ABA")
argss=("" "up up none" "up down near" "down up mid" "down down far" "-15 0.060" "-15 0.120" "-2 0.060" "-2 0.120" "2 0.060" "2 0.120" "15 0.060" "15 0.120")
sound=${sounds[SLURM_ARRAY_TASK_ID]}
args=${argss[SLURM_ARRAY_TASK_ID]}
echo "name of demo: $sound"
echo "args to use: $args"

cd /om2/user/mcusi/gen-bayesian-auditory-scenes/
source activate gentle
export JULIA_PROJECT=/om2/user/mcusi/gen-bayesian-auditory-scenes/
../julia-1.1.1/bin/julia ./tools/pinfer.jl $sound $args
echo "Done! :)"