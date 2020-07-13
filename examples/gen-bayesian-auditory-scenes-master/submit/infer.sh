#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --array=0-23

sounds=(1A 1i 1ii 1iii 1iv 1v 2i 2ii 2iii 2iv 2v 2vi b45_small df-15_dt150 df-2_dt140 df-5_dt80 homophonic-gradual homophonic-sudden nature22_0 stdup_compdown_f1030 stdup_compdown_f1460 stdup_compdown_f590 stdup_compdown_fnone toneContinuity-basic)

fn=${sounds[SLURM_ARRAY_TASK_ID]}
echo "name of observation: $fn"

cd /om2/user/mcusi/gen-bayesian-auditory-scenes/
source activate gentle
export JULIA_PROJECT=/om2/user/mcusi/gen-bayesian-auditory-scenes/
../julia-1.1.1/bin/julia ./inference/infer.jl $fn guide
