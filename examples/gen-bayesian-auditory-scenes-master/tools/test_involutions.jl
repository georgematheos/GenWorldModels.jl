using WAV;
using Gen;
import Random;
include("../model/model.jl");
include("../inference/proposals.jl")
include("../model/gammatonegram.jl");
include("../model/time_helpers.jl");
include("../inference/routine.jl");
include("../inference/initialization.jl");

function test_involution_fixed_randomness_args(trace, randomness, randomness_args, involution)

    # sample from the randomness
    (fwd_choices, fwd_score, fwd_ret) = propose(randomness, (trace, randomness_args...))
    # run the involution
    (new_trace, bwd_choices, weight) = involution(trace, fwd_choices, fwd_ret, randomness_args)
    (new_randomness_trace, _) = generate(randomness, (new_trace,randomness_args...), bwd_choices)
    # could check: get_choices(new_randomness_trace) == bwd_choices
    new_fwd_ret = get_retval(new_randomness_trace)

    # run the involution again
    (trace_round_trip, fwd_choices_round_trip, reverse_weight) = involution(new_trace, bwd_choices, new_fwd_ret, randomness_args)

    # check the weight
    @assert(isapprox(reverse_weight, -weight), "isapprox(reverse_weight, -weight): $reverse_weight =/= -$weight")
    #plot_gtg(get_retval(trace_round_trip)[1] - get_retval(trace)[1], get_args(trace)[2], get_args(trace)[end-1]/2.)
    
    @assert trace[:scene] == trace_round_trip[:scene]
    #plot_gtg(trace[:scene] - trace_round_trip[:scene], get_args(trace)[2], get_args(trace)[end-1]/2.)
    s1 = project(trace, select(:scene));  s2 = project(trace_round_trip, select(:scene));
    @assert(s1 == s2, "Score-- scene $s1 =/= $s2")
    @assert(trace[:n_sources] == trace_round_trip[:n_sources], ["n_sources: ", trace[:n_sources], " =/= ", trace_round_trip[:n_sources]])
    s1 = project(trace, select(:n_sources));  s2 = project(trace_round_trip, select(:n_sources));
    @assert(s1 == s2, "Score-- n_sources $s1 =/= $s2")
    for source_idx = 1:trace[:n_sources]
        
        source_trace = get_submap(get_choices(trace), :source => source_idx);
        source_trace_round_trip = get_submap(get_choices(trace_round_trip), :source => source_idx)
        @assert(source_trace[:n_elements] == source_trace_round_trip[:n_elements],["n_elements: ", source_trace[:n_elements]," =/= ",source_trace_round_trip[:n_elements]])
        s1 = project(trace, select(:source => source_idx => :n_elements));  s2 = project(trace_round_trip, select(:source => source_idx => :n_elements));
        @assert(s1 == s2, "Score-- n_elements $s1 =/= $s2")
            
        @assert(source_trace[:source_type] == source_trace_round_trip[:source_type],["source_type: ", source_trace[:source_type]," =/= ",source_trace_round_trip[:source_type]])
        s1 = project(trace, select(:source => source_idx => :source_type));  s2 = project(trace_round_trip, select(:source => source_idx => :source_type));
        @assert(s1 == s2, "Score-- source_type $s1 =/= $s2")
        
        if source_trace[:source_type]
            for a = [:mu, :sigma, :scale, :noise]
                @assert(source_trace[:erb => a] == source_trace_round_trip[:erb => a],["erb $a: ", source_trace[:erb => a]," =/= ",source_trace_round_trip[:erb => a]])
                s1 = project(trace, select(:source => source_idx => :erb => a));  s2 = project(trace_round_trip, select(:source => source_idx => :erb => a));
            @assert(s1 == s2, "Score-- erb $a $s1 =/= $s2")
            end
        end 
        for a = [:mu, :sigma, :noise]
            @assert(source_trace[:amp => a] == source_trace_round_trip[:amp => a],["[:amp => a]: ", source_trace[:amp => a]," =/= ",source_trace_round_trip[:amp => a]])
            s1 = project(trace, select(:source => source_idx => :amp => a));  s2 = project(trace_round_trip, select(:source => source_idx => :amp => a));
            @assert(s1 == s2, "Score-- amp $a $s1 =/= $s2")
        end

        scale_list = source_trace[:source_type] ? [:scale] : [:scale_t, :scale_f]
        for a = scale_list
            @assert(source_trace[:amp => a] == source_trace_round_trip[:amp => a],["[:amp => a]: ", source_trace[:amp => a]," =/= ",source_trace_round_trip[:amp => a]])
            s1 = project(trace, select(:source => source_idx => :amp => a));  s2 = project(trace_round_trip, select(:source => source_idx => :amp => a));
            @assert(s1 == s2, "Score-- amp $a $s1 =/= $s2")
        end
        
        element_as = source_trace[:source_type] ? [:wait, :dur_minus_min, :erb, :amp] : [:wait, :dur_minus_min, :amp]
        
        for element_idx = 1:source_trace[:n_elements]
            
            for a = element_as
                
                @assert(isapprox(source_trace[(:element, element_idx) => a],source_trace_round_trip[(:element, element_idx) => a]), ["$a $element_idx: ", source_trace[(:element, element_idx) => a], " =/= ", source_trace_round_trip[(:element, element_idx) => a]])
                s1 = project(trace, select(:source => source_idx => (:element, element_idx) => a));  s2 = project(trace_round_trip, select(:source => source_idx => (:element, element_idx) => a));
                @assert( isapprox(s1,s2), "Score-- element $element_idx, $a : $s1 =/= $s2")
            end

        end
        
        
    end
    
end


function test_involution_move(model, randomness, involution, randomness_args_fn, args)

    # generate a random trace of the model
    trace = simulate(model, args) 
    # get args for ranadoness
    randomness_args_list = randomness_args_fn(trace)
    for randomness_args in randomness_args_list
        test_involution_fixed_randomness_args(trace, randomness, randomness_args, involution)
    end
    return "Successful test!"
end

function no_args_fn(trace)
    return [()]
end

function source_args_fn(trace)
    n_sources = trace[:n_sources]
    return [(source_id,) for source_id in 1:n_sources]
end

function element_args_fn(trace)
    args = [];
    for source_id = 1:trace[:n_sources]
        for element_idx = 1:trace[:source => source_id => :n_elements]
            push!(args, (element_idx, source_id,))
        end
    end
    return args
end

                    
function gp_args_fn(trace)
        
    n_max, scene_duration, wts, steps, audio_sr, annealing_noise = get_args(trace)
        
    args = []; step_size = 10;
    for source_id = 1:trace[:n_sources]
        source_trace = get_submap(get_choices(trace),:source=>source_id)
        source_timings = absolute_timing(source_trace, steps["min"])
        gp_types = source_trace[:source_type] ? [:erb, :amp] : [:amp]
        for element_idx = 1:source_trace[:n_elements]
            onset = source_timings[element_idx][1]; offset = source_timings[element_idx][2]
            if source_trace[:source_type]
                x = get_element_gp_times([onset, offset], steps["t"])
            else
                x, _, _ = get_gp_spectrotemporal([onset, offset], steps, audio_sr)
            end
            k = length(x)
            for var_type in gp_types
                for i = 1:step_size:k
                    update_idxs = Int.(i : min(k, Int(i+step_size-1)))
                    push!(args, (element_idx, update_idxs, var_type, source_id,))
                end
                push!(args, (element_idx, 1:k, var_type, source_id,))
            end
        end
    end
    return args    
end

# mh(trace, swap_sources_randomness, (), swap_sources_involution)
# mh(trace, n_randomness, (source_id,), n_involution)
# mh(trace, wait_randomness, (tone_idx,source_id,), wait_involution)
# mh(trace, duration_randomness, (tone_idx,source_id,), duration_involution)
# mh(trace, gp_randomness, (tone_idx, update_idxs,"erb",source_id), gp_involution)
# mh(trace, gp_randomness, (tone_idx, update_idxs, "amp",source_id), gp_involution)
# mh(trace, gp_randomness, (tone_idx, 1:k, "erb", source_id), gp_involution)
# mh(trace, gp_randomness, (tone_idx, 1:k, "amp", source_id), gp_involution)
# mh(trace, sm_randomness, (source_id,), sm_involution)
# mh(trace, switch_randomness, (), switch_involution) 

println("Defining args...")
max_elements=10; duration=1.0; audio_sr=20000; steps = Dict(:t=>0.020, :min=> 0.020, :f=>4); annealing_noise=-1
wts, f = gtg_weights(audio_sr)
args=(max_elements, duration, wts, steps, audio_sr, annealing_noise)

println("Testing swap sources...")
success = test_involution_move(generate_scene, swap_sources_randomness, swap_sources_involution, no_args_fn,args)
println("Swap sources: ", success)
println("Testing n...")
success = test_involution_move(generate_scene, n_randomness, n_involution, source_args_fn,args)
println("n: ", success)
println("Testing wait...")
success = test_involution_move(generate_scene, wait_randomness, wait_involution, element_args_fn,args)
println("Wait: ", success)
println("Testing duration...")
success = test_involution_move(generate_scene, duration_randomness, duration_involution, element_args_fn,args)
println("Duration: ", success)
println("Testing GP...")
success = test_involution_move(generate_scene, gp_randomness, gp_involution, gp_args_fn,args)
println("GP: ", success)
println("Testing switch...")
for i = 1:100
    Random.seed!(i)
    success = test_involution_move(generate_scene, switch_randomness, switch_involution, no_args_fn,args)
    println("$i switch: ", success)
end
println("Testing split/merge...")
for i = 1:100
    Random.seed!(i)
    success = test_involution_move(generate_scene, sm_randomness, sm_involution, source_args_fn,args)
    println("$i split/merge: ", success)
end