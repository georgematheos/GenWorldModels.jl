using WAV;
using Gen;
using Printf;
import Random;
include("../tools/plotting.jl");
include("../inference/proposals.jl");
include("../inference/routine.jl");
include("../inference/initialization.jl");
include("../model/model.jl");
include("../model/gammatonegram.jl");
include("../model/time_helpers.jl");

println("Loading in command-line args...")
demo_name = ARGS[1]
expt_name = ARGS[2]
params_name = ARGS[3]

println("Setting random seed...")
random_seed = abs(rand(Int,1)[1])
Random.seed!(random_seed)

#Start experiment directory
PyPlot.ioff()
searchdir(path,key) = filter(x->occursin(key,x), readdir(path))
list_of_expts = searchdir("./sounds/",string(demo_name,"_",expt_name,"_"))
iter = length(list_of_expts) + 1
itername = @sprintf("%02d",iter); 
save_loc = string("./sounds/",demo_name,"_",expt_name,"_$itername/");
print("Making folder: "); println(save_loc)
mkdir(save_loc)
d = Dict()
d["seed"] = random_seed
open(string(save_loc, "random_seed.json"),"w") do f
    JSON.print(f, d)
end

println("Defining model parameters...")
## Model parameters
#not yet set up to do anything other than max n
source_params, steps, gtg_params, obs_noise = include(string("../params/",params_name,".jl"))
source_params["dream"]=false
cp(string("./params/",params_name,".jl"), string(save_loc,"params_",params_name,".jl"))

## Inference parameters
println("Defining inference parameters...")
n_steps = 2
n_likelihood_samples = 0
println("Loading in sound...")
## Inference
#Load in sound observation and its neural network guide distribution
demo_gram, wts, scene_duration, audio_sr = load_sound(demo_name, gtg_params)
#demo_guide = read_guide(demo_name)

println("Doing inference...")
#Sample initialization from the guide distribution 
args = (source_params, Float64(scene_duration), wts, steps, Int(audio_sr), obs_noise, gtg_params)
constraints = choicemap()
constraints[:scene] = demo_gram;
init_trace, = n_likelihood_samples > 0 ? init_from_likelihood(n_likelihood_samples, demo_guide, demo_gram, args) : generate(generate_scene, args, constraints);

#do approximate posterior inference
traces, proposal_counts = run_inference(init_trace, demo_gram, n_steps, save_loc=save_loc);
println("Inference complete!")

println("Saving our results.")
## Save results
# Show frequency of proposal acceptance for each proposal type
plot_proposals(proposal_counts; save_loc=save_loc)

