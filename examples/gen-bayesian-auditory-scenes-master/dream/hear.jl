include("../dream/dreaming.jl")

println("Loading in command-line args...")
#Load in command-line arguments
dataset_name = ARGS[1]
println("Dataset name: ", dataset_name)

println("Starting to generate gammatonegrams for demos...")
hearing(dataset_name)