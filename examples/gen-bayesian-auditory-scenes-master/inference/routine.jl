using Gen;
include("../tools/plotting.jl")
include("../inference/proposals.jl")

function make_annealing_schedule(slope, midpoint, height, base, cutoff)
    function annealing_schedule(x)
        sigmoid_input = -slope * (x - midpoint)
        noise = height*( 1/(1 + exp(-sigmoid_input)) ) + base
        return x > cutoff ? base : noise
    end
    return annealing_schedule
end

function increment_proposal_counts(k, accepted, proposal_counts)
    if ~(k in keys(proposal_counts))
        proposal_counts[k] = Dict("a"=>0, "tot"=>0)
    end
    proposal_counts[k]["a"] += Int(accepted);
    proposal_counts[k]["tot"] += 1;
    return proposal_counts
end
    

function mcmc_update(trace, proposal_counts)

    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    
    #swap to the end
    #change by arbitrary number
    #trace, accepted = mh(trace, select(:n_sources))
    trace, accepted = mh(trace, change_n_sources, ())
    proposal_counts = increment_proposal_counts("nsrc", accepted, proposal_counts)

    trace, accepted = mh(trace, swap_sources_randomness, (), swap_sources_involution)
    @assert accepted
    
    for source_id = 1:trace[:n_sources]
        
        #add or remvoe a tone
        trace, accepted = mh(trace, n_randomness, (source_id,), n_involution)
        proposal_counts = increment_proposal_counts("nelem", accepted, proposal_counts)
        
        source_type = source_params["types"][trace[:source => source_id => :source_type]]
        gp_types = source_type == "tone" || source_type == "harmonic" ? [:erb, :amp] : [:amp]
        for gp_type = gp_types
            source_attributes = keys( gp_type === :erb ? source_params["gp"][String(gp_type)] : (source_type == "tone" ? source_params["gp"][String(gp_type)]["1D"] : source_params["gp"][String(gp_type)]["2D"]) )
            for sa = source_attributes
                trace, accepted = mh(trace, select(:source => source_id => gp_type => Symbol(sa) ))
                proposal_counts = increment_proposal_counts("gp", accepted, proposal_counts)
            end
        end
        for tp_type = [:wait,:dur_minus_min]
            for sa = [:a, :mu]
                trace, accepted = mh(trace, select(:source => source_id => tp_type => sa))
                proposal_counts = increment_proposal_counts("tp", accepted, proposal_counts)
            end
        end

        #elements
        nt = get_choices(trace)[:source => source_id => :n_elements]
        for element_idx = 1:nt
            
            trace, accepted = mh(trace, wait_randomness, (element_idx,source_id,), wait_involution)
            proposal_counts = increment_proposal_counts("wait", accepted, proposal_counts)

            
            trace, accepted = mh(trace, duration_randomness, (element_idx,source_id,), duration_involution)
            proposal_counts = increment_proposal_counts("dur", accepted, proposal_counts)

            source_trace = get_submap(get_choices(trace),:source=>source_id)
            old_abs_timings = absolute_timing(source_trace, steps["min"])
            onset = old_abs_timings[element_idx][1]; offset = old_abs_timings[element_idx][2]
            
            for gp_type in gp_types

                if gp_type == :erb || (gp_type == :amp && source_type == "tone")
                    x = get_element_gp_times([onset, offset], steps["t"])     
                else
                    x, _, _ = get_gp_spectrotemporal([onset, offset], steps, audio_sr)                         
                end

                k = length(x); step_size = 20;
                for i = 1:step_size:k
                    update_idxs = Int.(i : min(k, Int(i+step_size-1)))
                    trace, accepted = mh(trace, gp_randomness, (element_idx, update_idxs, gp_type,source_id), gp_involution)
                    proposal_counts = increment_proposal_counts("gplocal", accepted, proposal_counts)

                end

                trace, accepted = mh(trace, gp_randomness, (element_idx, 1:k, gp_type, source_id), gp_involution)
                proposal_counts = increment_proposal_counts("gpfull", accepted, proposal_counts)

            end

        end
        #check if we should split or merge some random pair of consecutive tones
        #Could run this proposal more than once per cycle
        trace, accepted = mh(trace, sm_randomness, (source_id,), sm_involution)
        proposal_counts = increment_proposal_counts("splmrg", accepted, proposal_counts)

        if length(source_params["types"]) == 3 && source_params["enable_switch"]
            trace, accepted = mh(trace, type_randomness, (source_id,), type_involution)
            proposal_counts = increment_proposal_counts("type", accepted, proposal_counts)
        end
        
        ## TODO: Re-propose source latents all together based on amortized proposal 
                            
    end

    #check if we should move a random tone into a different stream
    for k = 1
        trace, accepted = mh(trace, switch_randomness, (), switch_involution) 
        proposal_counts = increment_proposal_counts("switch", accepted, proposal_counts)
    end
    
    if obs_noise["type"] == "rand"
        trace, accepted = mh(trace, select(:obs_noise))
        proposal_counts = increment_proposal_counts("obsnoise", accepted, proposal_counts)
    end
    
    return trace, proposal_counts

end

function run_inference(initial_trace, obs_gram, max_steps; save_loc="./", inference_function=mcmc_update)

    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(initial_trace)

    if obs_noise["type"] == "anneal"
        slope = 0.5; midpoint = 200; height = 0.7; base = 0.8; cutoff = 300
        annealing_schedule = make_annealing_schedule(slope, midpoint, height, base, cutoff)
    end

    ##get custom proposals here with "include("custom_proposals.jl")" or something like that.
    
    trace = initial_trace
    traces = [trace,]
    proposal_counts = Dict()
    for i=1:max_steps
        print("$i ")
        if obs_noise["type"] == "anneal"
            #compute annealing schedule
            annealing_noise = annealing_schedule(i)
            obs_noise = Dict("type"=>"anneal","val"=>annealing_noise)
            #update trace with new annealing noise
            (trace,_,_,_) = update(trace, (source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params), (NoChange(), NoChange(), NoChange(), NoChange(), NoChange(), NoChange(), UnknownChange(),NoChange(),), choicemap())
        end
        trace, proposal_counts = inference_function(trace, proposal_counts)
        push!(traces, trace)
        plot_sources(trace, obs_gram, i, save_loc=save_loc)
    end
    
    return traces, proposal_counts
    
end
                    