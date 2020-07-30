#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=mcdermott

start=`date +%s`

dataset="$1"
echo "Dataset: $dataset"
end_folder="$2"
echo "End2End folder: ${end_folder}"
cd /om2/user/mcusi/gen-bayesian-auditory-scenes/
echo "Current directory:"
echo `pwd`

source activate gentle
export JULIA_PROJECT=/om2/user/mcusi/gen-bayesian-auditory-scenes/

demos=`ls /om/user/mcusi/dcbasa/experiments/${end_folder}/guide/`
demo=`echo $demos | cut -d" " -f${SLURM_ARRAY_TASK_ID} | rev | cut --complement -d"_" -f1 | rev` #gives you the demo name 
echo "Looking at demo $demo."

../julia-1.1.1/bin/julia ./inference/viewguide.jl $demo ${end_folder} ${dataset}

end=`date +%s`
echo "Duration: $((($(date +%s)-$start)/60)) minutes"