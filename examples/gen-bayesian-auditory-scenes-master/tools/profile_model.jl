using WAV


include("../model/model.jl");
#include("plotting.jl");
#include("proposals.jl");
include("../model/gammatonegram.jl");
include("../model/time_helpers.jl");
#include("inference_helpers.jl"); 
include("../inference/initialization.jl");

using Profile

max_tones = 16; 
tstep = 0.020
mindur = 0.020
    
#GP parameters -- eventually should be hyperpriors
erb = Dict(:sigma => 3, :scale => 0.5, :noise => 1)
amp = Dict(:sigma => 1, :scale => 0.5, :noise => 0.5)
get_kernel_params = make_get_kernel_params(erb, amp)

demo_name = "Track17X"
demo_gram, wts, scene_duration, audio_sr = load_sound(demo_name)
args = (max_tones, float(scene_duration), wts, mindur, tstep, Int(audio_sr), 0.1)

function do_test(n)
    for i=1:n
        trace = simulate(generate_scene, args)
    end
end

# run once to force Julia to precompile the code
do_test(10)

# now profile the precompiled code
Profile.init()
@profile do_test(1000)

# save profile results
li, lidict = Profile.retrieve()
using JLD
@save "profdata.jld" li lidict
