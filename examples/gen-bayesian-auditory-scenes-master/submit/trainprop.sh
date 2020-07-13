#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1

cd /om2/user/mcusi/gen-bayesian-auditory-scenes/

source /om/user/mcusi/dcbasa/environ.sh
singularity exec --nv --home $singularity_home --bind $singularity_bind $singularity_image bash -c "export LD_LIBRARY_PATH=$cupti:\$LD_LIBRARY_PATH; 
		export JULIA_PROJECT=/om2/user/mcusi/gen-bayesian-auditory-scenes/; 
		../julia-1.1.1/bin/julia train_proposals.jl"

echo "Done! :)"