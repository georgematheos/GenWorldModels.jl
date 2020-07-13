import Random;
using Printf;
using JSON
include("../tools/perfect_init.jl")

println("Loading in command-line args...")
demo_name = ARGS[1]
println("demo name is: $demo_name")

demoargs=[]; demofolder = ""
if demo_name == "tougas_bregman_1A"
    demofunc = tougas_bregman_1A
    demoargs = [] #empty
    demofolder = string(demofolder, demo_name,"-")
elseif demo_name == "bregman_rudnicky"
    demofunc = bregman_rudnicky
    standard=ARGS[2];println("Standard: $standard")#"up" or "down"
    comparison=ARGS[3];println("comparison: $comparison")#"up" or "down"
    captor=ARGS[4];println("captor: $captor")#"none","far","mid","near"
    demoargs = [standard, comparison, captor]
    demofolder = string(demofolder, demo_name , "-", standard, "_", comparison, "_", captor, "-")
elseif demo_name == "ABA"
    demofunc = ABA
    semitones=parse(Float64,ARGS[2]);println("semitones: $semitones") #-15 to +15
    spacing=parse(Float64,ARGS[3]);println("spacing: $spacing")#0.040s to 0.800s
    demoargs = [semitones,spacing]
    demofolder = string(demofolder, demo_name,"-",spacing,"_",semitones,"-")
end
demoargs = Tuple(demoargs)

println("Setting random seed...")
random_seed = abs(rand(Int,1)[1])
Random.seed!(random_seed)

searchdir(path,key) = filter(x->occursin(key,x), readdir(path))
list_of_expts = searchdir("../sounds/",demofolder)
iter = length(list_of_expts) + 1
itername = @sprintf("%02d",iter); 
save_loc = string("../sounds/",demofolder,"$itername/");
print("Making folder: "); println(save_loc)
mkdir(save_loc)
d = Dict()
d["seed"] = random_seed
open(string(save_loc, "random_seed.json"),"w") do f
    JSON.print(f, d)
end


trace = perfect_initialization(demofunc,demoargs);
scene_gram, t, scene_wave, source_waves, element_waves=get_retval(trace)
plot_sources(trace, scene_gram, 0; save_loc=save_loc)

n_inference_steps = 500
proposal_counts = Dict()

for i = 1:n_inference_steps
    print("$i ")
    
    global trace
    global accept_counts
    global proposal_counts
    global save_loc
    
    trace, proposal_counts = mcmc_update(trace, proposal_counts)
    scene_gram, t, scene_wave, source_waves, element_waves=get_retval(trace)
    plot_sources(trace, scene_gram, i; save_loc=save_loc)
end