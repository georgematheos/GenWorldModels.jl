using WAV;
using Gen;
using Printf;
import Random;
include("../tools/plotting.jl");
include("../inference/initialization.jl");
include("../model/extra_distributions.jl")
include("../model/model.jl")

println("Loading in command-line args...")
demo_name = ARGS[1]
guide_location = ARGS[2]
dataset_name = ARGS[3]
params_name = dataset_name

println("Setting random seed...")
random_seed = 1#abs(rand(Int,1)[1])
Random.seed!(random_seed)

PyPlot.ioff()
save_loc = string("../sounds/",demo_name,"-",guide_location,"-",params_name,"_guide/");
print("Making folder: "); println(save_loc)
mkdir(save_loc)
d = Dict()
d["seed"] = random_seed
open(string(save_loc, "random_seed.json"),"w") do f
    JSON.print(f, d)
end
dream_path="/om2/user/mcusi/gen-bayesian-auditory-scenes/dream/"
dream_path = string(dream_path, dataset_name, "_00/")

println("Defining model parameters...")
## Model parameters
#not yet set up to do anything other than max n
params_location = string(dream_path,"params_",params_name,".jl")
cp(params_location, string(save_loc,"params_",params_name,".jl"))
source_params, steps, gtg_params, obs_noise = include(string(save_loc,"params_",params_name,".jl"))
source_params["dream"]=false

println("Loading in sound...")
## Inference
#Load in sound observation and its neural network guide distribution
demo_gram, wts, scene_duration, audio_sr = load_demo_sound(demo_name, gtg_params)
demo_guide = read_guide(demo_name; demo_folder=string("/om/user/mcusi/dcbasa/experiments/",guide_location,"/guide/"))

println("Doing inference...")
#Sample initialization from the guide distribution 
args = (source_params, float(scene_duration), wts, steps, Int(audio_sr), obs_noise, gtg_params)
constraints = choicemap()
constraints[:scene] = demo_gram;
n_samples = 10
init_traces = sample_inits(n_samples, demo_guide, demo_gram, args) 
for i = 1:n_samples
	plot_sources(init_traces[i], demo_gram, i; colors="Blues", save_loc=save_loc)
end
println("Complete!")
