include("../dream/dreaming.jl")

println("Loading in command-line args...")
#Load in command-line arguments
dataset_name = ARGS[1]
println("Dataset name: ", dataset_name)
params_name = ARGS[2]
println("Params file: ", params_name)
parallel_run_idx = ARGS[3]
println("Parallel run index: ", parallel_run_idx)
n_samples = parse(Int, ARGS[4])
println("Number of samples: ", n_samples);

#Need a new random seed everytime, in case slurm quits and restarts:
random_seed = abs(rand(Int,1)[1])
Random.seed!(random_seed) 
println("Random seed: ", random_seed) 

println("Starting to dream...")
dreaming(dataset_name, params_name, parallel_run_idx, n_samples; dream_path="../dreams/")