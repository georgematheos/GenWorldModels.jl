#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=mcdermott

start=$SECONDS

echo "Running autosubmit4.sh"

dataset="$1"
echo "Dataset: $dataset"
end_folder="$2"
echo "End2End folder: ${end_folder}"
cd /om2/user/mcusi/gen-bayesian-auditory-scenes/
echo "Current directory:"
echo `pwd`

ndemos=$(ls /om/user/mcusi/dcbasa/experiments/${end_folder}/guide/ | wc -l)
echo "There are $ndemos demos."

viewinit=$(sbatch --parsable --array=1-$ndemos ./submit/autoviewguide.sh ${dataset} ${end_folder})
echo "generating inits from bayesian model ID: ${viewinit}"

duration=$(( SECONDS - start ))
echo 'autosubmit4- Time taken in seconds:'
echo $duration