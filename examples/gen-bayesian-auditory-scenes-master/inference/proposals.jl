using Gen;
using Statistics: mean, std;
include("../model/model.jl")
include("../model/time_helpers.jl")
include("../model/gaussian_helpers.jl")

### Custom Proposals

@gen function change_n_sources(trace)
   
   source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
   max_sources = source_params["n_sources"]["val"]
   lower_bound = max(1, trace[:n_sources]-1)
   upper_bound = min(trace[:n_sources]+1, max_sources)
   new_n = @trace(uniform_discrete(lower_bound, upper_bound), :n_sources)
   if new_n > trace[:n_sources]
        @trace( generate_source(source_params, scene_duration, steps, audio_sr, gtg_params), :source => new_n)
   end
end

### Involutions

## ADD OR REMOVE A TONE
@gen function n_randomness(trace, source_id)
    
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    max_elements = source_params["n_elements"]["val"]
    #get old parameters
    old_choices = get_submap(get_choices(trace), :source => source_id)
    old_n_elements = old_choices[:n_elements]
    old_abs_timings = absolute_timing(old_choices, steps["min"])
    source_type = source_params["types"][old_choices[:source_type]]
    
    #Add or delete element?
    if old_n_elements == 1
        #Cannot get rid of a element if it's the only one in a stream
        p_add = 1
    elseif 1 < old_n_elements < max_elements
        p_add = 1/2
    elseif old_n_elements == max_elements
        #Cannot add a element if that stream already has the max number of elements
        p_add = 0 
    end
    add_element = @trace(bernoulli(p_add), :add_element)
    new_n_elements = old_n_elements + (add_element ? 1 : -1)

    if add_element
        
        #Adding a new element, need to choose its index and sample its new features
        #If you choose an existing spot, then all the ones later slide to the later position.  
        add_element_idx = @trace(uniform_discrete(1, new_n_elements), :add_element_idx)

        #Need to sample onset/offset in the gap between existing elements:
        #For A = lower limit on onset, and B = upper limit on offset
        #(A < add_onset < add_offset < B) && (add_offset - add_onset >= steps["min"])
        if add_element_idx == 1
            A = 0 #start of sound
            B = old_abs_timings[1][1] #Onset of the currently first element
        elseif 1 < add_element_idx < new_n_elements
            A = old_abs_timings[add_element_idx - 1][2] #Offset of element before it
            B = old_abs_timings[add_element_idx][1] #Onset that USED to have its same index
        elseif add_element_idx == new_n_elements #Comes last
            #No upper limit on offset: will need to use exponential rather than uniform distribution to sample
            A = old_abs_timings[end][2]
        end

        #check conditions and sample randomness
        if add_element_idx < new_n_elements
            if (B-steps["min"])-A < 0
                return "abort"
            end
        end
        add_wait = add_element_idx == new_n_elements ? @trace(exponential(1), :add_wait) : @trace(uniform(0, (B-steps["min"]) - A), :add_wait)
        add_onset = A + add_wait
        if add_element_idx < new_n_elements
            if B-add_onset < steps["min"]
                return "abort"
            end
        end
        add_duration = add_element_idx == new_n_elements ? (@trace(exponential(1), :add_durminusmin) + steps["min"]) : @trace(uniform(steps["min"], B - add_onset), :add_duration)
        add_offset = add_onset + add_duration
        add_timing = [add_onset, add_offset]

        #Collect the GPs associated with all other elements in order to condition the new GP samples on them
        add_t = Dict()
        if source_type == "tone"           
            old_gps = Dict(:erb=>[], :amp=>[]);
            old_gps_points = Dict(:erb=>[], :amp=>[])
            for i = 1:old_n_elements
                append!(old_gps_points[:erb], get_element_gp_times(old_abs_timings[i], steps["t"]))
                append!(old_gps_points[:amp], get_element_gp_times(old_abs_timings[i], steps["t"]))
                append!(old_gps[:erb], old_choices[(:element, i) => :erb])
                append!(old_gps[:amp], old_choices[(:element, i) => :amp])
            end
            add_t[:erb] = get_element_gp_times(add_timing, steps["t"])
            add_t[:amp] = get_element_gp_times(add_timing, steps["t"])
        elseif source_type == "noise"
            old_gps = Dict(:amp=>[]);
            old_gps_points = Dict(:amp=>[])
            for i = 1:old_n_elements 
                append!(old_gps_points[:amp], get_gp_spectrotemporal(old_abs_timings[i], steps, audio_sr)[1])
                append!(old_gps[:amp], old_choices[(:element, i) => :amp])
            end
            add_t[:amp] = get_gp_spectrotemporal(add_timing, steps, audio_sr)[1]
        elseif source_type == "harmonic"
            old_gps = Dict(:erb=>[], :amp=>[]);
            old_gps_points = Dict(:erb=>[], :amp=>[])
            for i = 1:old_n_elements
                append!(old_gps_points[:erb], get_element_gp_times(old_abs_timings[i], steps["t"]))
                append!(old_gps_points[:amp], get_gp_spectrotemporal(old_abs_timings[i], steps, audio_sr)[1])
                append!(old_gps[:erb], old_choices[(:element, i) => :erb])
                append!(old_gps[:amp], old_choices[(:element, i) => :amp])
            end
            add_t[:erb]=get_element_gp_times(add_timing, steps["t"])
            add_t[:amp]=get_gp_spectrotemporal(add_timing, steps, audio_sr)[1]
        end
                                                 
        #Sample conditional GPs        
        for gp_type = keys(old_gps)
            mu, cov = get_cond_mu_cov(add_t[gp_type], old_gps_points[gp_type], old_gps[gp_type], get_submap(old_choices, gp_type))
            @trace(mvnormal(mu, cov), gp_type) 
        end
        
        return add_offset
        
    else
        #Taking away an old element, need to choose which one
        ps = Vector(ones(old_n_elements)/old_n_elements)
        remove_element_idx = @trace(categorical(ps), :remove_element_idx)
    end                                    
end




function n_involution(trace, fwd_choices, fwd_ret, proposal_args)
    
    if fwd_ret == "abort"
        return trace, fwd_choices, 0
    end
                                    
    #we need to specify how to go backwards
    #and how to construct the new trace
    bwd_choices = choicemap()
    new_choices = choicemap()
    source_id = proposal_args[1]
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    max_elements = source_params["n_elements"]["val"]
    old_choices = get_submap(get_choices(trace), :source => source_id)
    source_type = source_params["types"][old_choices[:source_type]]
    gp_types = source_type == "tone" || source_type == "harmonic" ? [:erb, :amp] : [:amp]
    element_attributes = vcat(gp_types, [:dur_minus_min]); 
    
    new_n_elements = fwd_choices[:add_element] ? old_choices[:n_elements] + 1 : old_choices[:n_elements] - 1
    new_choices[:source => source_id => :n_elements] = new_n_elements
    old_n_elements = old_choices[:n_elements]
    bwd_choices[:add_element] = !fwd_choices[:add_element]
    old_abs_timings = absolute_timing(old_choices, steps["min"])
                                                                            
    if fwd_choices[:add_element]
                
        ##Add_element
        #Add in new timing info 
        add_element_idx = fwd_choices[:add_element_idx] 
        new_dur_minus_min = add_element_idx == new_n_elements ? fwd_choices[:add_durminusmin] : (fwd_choices[:add_duration] - steps["min"])
        new_duration = add_element_idx == new_n_elements ? (fwd_choices[:add_durminusmin] + steps["min"]) : fwd_choices[:add_duration]                                              
        new_choices[:source => source_id => (:element, add_element_idx) => :wait] = fwd_choices[:add_wait]
        new_choices[:source => source_id => (:element, add_element_idx) => :dur_minus_min] = new_dur_minus_min
        #Add in new gps 
        for gp_type = gp_types
            new_choices[:source => source_id =>(:element, add_element_idx) => gp_type] = fwd_choices[gp_type]
        end
                        
        if add_element_idx != new_n_elements
            #If there are any elements after the add_element: 
            #Need to change the "wait" of the element immediately after the add_element, in order to retain the same onset
            new_choices[:source => source_id => (:element, add_element_idx + 1) => :wait] = old_abs_timings[add_element_idx][1] - fwd_ret #add_element offset is returned from randomness function
            for a in element_attributes
                new_choices[:source => source_id => (:element, add_element_idx + 1) => a] = old_choices[(:element, add_element_idx) => a]
            end
            #All elements after (the element immediately following the add_telement) are simply shifted in index
            old_idxs = add_element_idx + 1 == new_n_elements ? [] : add_element_idx+1:new_n_elements-1
            for old_element_idx = old_idxs
                set_submap!(new_choices, :source => source_id => (:element, old_element_idx+1), get_submap(old_choices, (:element,old_element_idx)))                 
            end
        end
                                                            
        bwd_choices[:remove_element_idx] = add_element_idx
                    
    else
        
        #Shift all elements after the element removed
        remove_idx = fwd_choices[:remove_element_idx]
        if remove_idx < old_n_elements
            #Adjust the wait time of the element following the one that was removed 
            last_offset = remove_idx > 1 ? old_abs_timings[remove_idx - 1][2] : 0
            new_choices[:source => source_id => (:element, remove_idx) => :wait] = old_abs_timings[remove_idx+1][1] - last_offset 
            for a in element_attributes
                new_choices[:source => source_id => (:element, remove_idx) => a] = old_choices[(:element, remove_idx+1) => a]
            end
            #All the elements after the one immediately following the remove_element, must shift their indices
            old_idxs = remove_idx == new_n_elements ? [] : (remove_idx + 2):old_n_elements
            for old_element_idx = old_idxs
                set_submap!(new_choices, :source => source_id => (:element, old_element_idx-1), get_submap(old_choices, (:element,old_element_idx)))
            end
        end

        #Define backwards choice by putting remove_elementback in                                                                
        bwd_choices[:add_element_idx] = remove_idx        
        bwd_choices[:add_wait] = old_choices[(:element, remove_idx) => :wait]  
        if remove_idx < old_n_elements
            bwd_choices[:add_duration] = old_choices[(:element, remove_idx) => :dur_minus_min] + steps["min"]
        else
            bwd_choices[:add_durminusmin] = old_choices[(:element, remove_idx) => :dur_minus_min]
        end
        for gp_type = gp_types
            bwd_choices[gp_type] = old_choices[(:element, remove_idx) => gp_type]
        end
                                                                                                    
    end
    
    new_trace, weight = update(trace, get_args(trace), (), new_choices)
    return new_trace, bwd_choices, weight

end


## CHANGE ONSET ONLY
@gen function wait_randomness(trace, element_idx, source_id)
    
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    max_elements = source_params["n_elements"]["val"]
    #get old parameters
    old_abs_timings = absolute_timing(get_submap(get_choices(trace), :source => source_id), steps["min"])
    old_choices = get_submap(get_choices(trace), :source => source_id => (:element, element_idx))
    old_onset = old_abs_timings[element_idx][1]; old_offset = old_abs_timings[element_idx][2]; 
    
    #The new wait value can go from 0 to the largest value that leaves the element with duration at least steps["min"]
    W = old_choices[:wait] + old_choices[:dur_minus_min] - steps["min"]
    if W < 0
        return "abort"
    end
    new_wait = @trace(uniform(0, W), :wait)   
    
    #Figure out whether the new wait value requires new GP values to be sampled 
    new_onset = (element_idx == 1 ? 0 : old_abs_timings[element_idx-1][2]) + new_wait
    old_gp_timings = get_element_gp_times(old_abs_timings[element_idx], steps["t"])
    new_gp_timings = get_element_gp_times([new_onset, old_abs_timings[element_idx][2]], steps["t"])
    extra_dims = length(new_gp_timings) - length(old_gp_timings)
    
    #sample randomness if new gp elements are needed
    if extra_dims > 0
        
        source_type = source_params["types"][trace[:source => source_id => :source_type]]
        if source_type == "tone" #"tone"
            new_t = new_gp_timings[1:extra_dims]
            for gp_type = [:erb, :amp]
                gp_latents = get_submap(get_choices(trace), :source => source_id => gp_type)
                new_gp_latents = Dict(:mu => mean(old_choices[gp_type]), :scale => gp_latents[:scale], :sigma => gp_latents[:sigma], :epsilon => gp_latents[:epsilon])
                mu, cov = get_cond_mu_cov(new_t, old_gp_timings, old_choices[gp_type], new_gp_latents)
                @trace(mvnormal(mu, cov), gp_type) 
            end
        elseif source_type == "noise" #noise
            tfs, _, fs = get_gp_spectrotemporal([new_onset, old_abs_timings[element_idx][2]], steps, audio_sr)
            #extra dims is in time, need to take the frequencies with that
            new_tfs = tfs[1:extra_dims*length(fs)]
            old_tfs = tfs[(extra_dims*length(fs)+1):end]
                        
            gp_latents = get_submap(get_choices(trace), :source => source_id => :amp)
            new_gp_latents = Dict(:mu => mean(old_choices[:amp]), :scale_t => gp_latents[:scale_t], :scale_f => gp_latents[:scale_f], :sigma => gp_latents[:sigma], :epsilon => gp_latents[:epsilon])
            
            
            mu, cov = get_cond_mu_cov(new_tfs, old_tfs, old_choices[:amp], new_gp_latents)
            @trace(mvnormal(mu, cov), :amp)   
        elseif source_type == "harmonic"

            #Fundamental frequency
            new_t = new_gp_timings[1:extra_dims]
            gp_latents = get_submap(get_choices(trace), :source => source_id => :erb)
            new_gp_latents = Dict(:mu => mean(old_choices[:erb]), :scale => gp_latents[:scale], :sigma => gp_latents[:sigma], :epsilon => gp_latents[:epsilon])
            mu, cov = get_cond_mu_cov(new_t, old_gp_timings, old_choices[:erb], new_gp_latents)
            @trace(mvnormal(mu, cov), :erb) 

            #Amplitude 
            tfs, _, fs = get_gp_spectrotemporal([new_onset, old_abs_timings[element_idx][2]], steps, audio_sr)
            #extra dims is in time, need to take the frequencies with that
            new_tfs = tfs[1:extra_dims*length(fs)]
            old_tfs = tfs[(extra_dims*length(fs)+1):end]
                        
            gp_latents = get_submap(get_choices(trace), :source => source_id => :amp)
            new_gp_latents = Dict(:mu => mean(old_choices[:amp]), :scale_t => gp_latents[:scale_t], :scale_f => gp_latents[:scale_f], :sigma => gp_latents[:sigma], :epsilon => gp_latents[:epsilon])
            
            mu, cov = get_cond_mu_cov(new_tfs, old_tfs, old_choices[:amp], new_gp_latents)
            @trace(mvnormal(mu, cov), :amp)   

        end

    end
    
    return new_onset, length(old_gp_timings), length(new_gp_timings)
    
end


function wait_involution(trace, fwd_choices, fwd_ret, proposal_args)
                            
    if fwd_ret == "abort"
        return trace, fwd_choices, 0
    end
                            
    #we need to specify how to go backwards
    #and how to construct the new trace
    bwd_choices = choicemap()
    new_choices = choicemap()
    element_idx = proposal_args[1]; source_id = proposal_args[2];
    new_onset = fwd_ret[1]; old_k = fwd_ret[2]; new_k = fwd_ret[3];
    
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    max_elements = source_params["n_elements"]["val"]
    old_choices = get_submap(get_choices(trace),:source => source_id => (:element, element_idx))
    
    ##Set new timings for (:element, element_idx) 
    #Wait
    new_choices[:source => source_id => (:element, element_idx) => :wait] = fwd_choices[:wait]
    #Duration (we want the offset to stay the same given the new proposal)
    #wait_diff < 0 means new wait is smaller than old wait, so duration of element must become longer
    wait_diff = fwd_choices[:wait] - old_choices[:wait]
    new_choices[:source => source_id => (:element, element_idx) => :dur_minus_min] = old_choices[:dur_minus_min] - wait_diff 
    bwd_choices[:wait] = old_choices[:wait]
    
    source_type = source_params["types"][trace[:source => source_id => :source_type]]
    gp_types = source_type == "tone" || source_type == "harmonic" ? [:erb,:amp] : [:amp]
    ##Set new gps for (:element, element_idx)
    if new_k > old_k
        for gp_type = gp_types
            new_gp = vcat(fwd_choices[gp_type], old_choices[gp_type])
            new_choices[:source => source_id => (:element, element_idx) => gp_type] = new_gp
        end   
    elseif new_k < old_k
        for gp_type = gp_types
            old_gp = old_choices[gp_type]
            lenf = (gp_type == :erb || (gp_type == :amp && source_type == "tone")) ? 1 : length(get_element_gp_freqs(audio_sr, steps))  
            bwd_choices[gp_type] = old_gp[1:lenf*(old_k-new_k)]
            new_choices[:source => source_id =>(:element, element_idx) => gp_type] = old_gp[lenf*(old_k-new_k)+1:end]
        end  
    end
    
    new_trace, weight = update(trace, get_args(trace), (), new_choices)
    return new_trace, bwd_choices, weight
end

## CHANGE DURATION ONLY

@gen function duration_randomness(trace, element_idx, source_id)
    
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    max_elements = source_params["n_elements"]["val"]
    #get old parameters
    old_source_choices = get_submap(get_choices(trace),:source => source_id); 
    old_choices = get_submap(get_choices(trace),:source => source_id => (:element, element_idx))
    old_abs_timings = absolute_timing(old_source_choices, steps["min"])
    old_onset = old_abs_timings[element_idx][1]; 

    #Duration can only be as short as mindur or increased up until the next element
    next_onset = element_idx == old_source_choices[:n_elements] ? scene_duration : old_abs_timings[element_idx + 1][2]
    W = next_onset - old_onset;
    if W < steps["min"]
        return "abort"
    end
    new_duration = @trace(uniform(steps["min"], W), :duration)   
    
    #Figure out whether new duration value requires new gp to be sampled
    new_offset = old_onset + new_duration
    old_gp_timings = get_element_gp_times(old_abs_timings[element_idx], steps["t"])
    new_gp_timings = get_element_gp_times([old_abs_timings[element_idx][1], new_offset], steps["t"])
    extra_dims = length(new_gp_timings) - length(old_gp_timings)
    
    source_type = source_params["types"][trace[:source => source_id => :source_type]]
    #sample randomness if new gp elements are needed
    if extra_dims > 0
        if source_type == "tone" #tone
            new_t = new_gp_timings[length(old_gp_timings)+1:end]
            for gp_type = [:erb, :amp]
                gp_latents = get_submap(get_choices(trace), :source => source_id => gp_type)
                new_gp_latents = Dict(:mu => mean(old_choices[gp_type]), :scale => gp_latents[:scale], :sigma => gp_latents[:sigma], :epsilon => gp_latents[:epsilon])
                mu, cov = get_cond_mu_cov(new_t, old_gp_timings, old_choices[gp_type], new_gp_latents)
                @trace(mvnormal(mu, cov), gp_type) 
            end
        elseif source_type == "noise"
            tfs, _, fs = get_gp_spectrotemporal([old_abs_timings[element_idx][1], new_offset], steps, audio_sr)  
            #extra dims is in time, need to take the frequencies with that
            old_tfs = tfs[1:length(old_gp_timings)*length(fs)]
            new_tfs = tfs[(length(old_gp_timings)*length(fs)+1):end]
            
            gp_latents = get_submap(get_choices(trace), :source => source_id => :amp)
            new_gp_latents = Dict(:mu => mean(old_choices[:amp]), :scale_t => gp_latents[:scale_t], :scale_f => gp_latents[:scale_f], :sigma => gp_latents[:sigma], :epsilon => gp_latents[:epsilon])
            
            mu, cov = get_cond_mu_cov(new_tfs, old_tfs, old_choices[:amp], new_gp_latents)
            @trace(mvnormal(mu, cov), :amp)   
        elseif source_type == "harmonic"

            #Fundamental frequency
            new_t = new_gp_timings[length(old_gp_timings)+1:end]
            gp_latents = get_submap(get_choices(trace), :source => source_id => :erb)
            new_gp_latents = Dict(:mu => mean(old_choices[:erb]), :scale => gp_latents[:scale], :sigma => gp_latents[:sigma], :epsilon => gp_latents[:epsilon])
            mu, cov = get_cond_mu_cov(new_t, old_gp_timings, old_choices[:erb], new_gp_latents)
            @trace(mvnormal(mu, cov), :erb) 

            #Filter
            tfs, _, fs = get_gp_spectrotemporal([old_abs_timings[element_idx][1], new_offset], steps, audio_sr)  
            #extra dims is in time, need to take the frequencies with that
            old_tfs = tfs[1:length(old_gp_timings)*length(fs)]
            new_tfs = tfs[(length(old_gp_timings)*length(fs)+1):end]
            
            gp_latents = get_submap(get_choices(trace), :source => source_id => :amp)
            new_gp_latents = Dict(:mu => mean(old_choices[:amp]), :scale_t => gp_latents[:scale_t], :scale_f => gp_latents[:scale_f], :sigma => gp_latents[:sigma], :epsilon => gp_latents[:epsilon])
            
            mu, cov = get_cond_mu_cov(new_tfs, old_tfs, old_choices[:amp], new_gp_latents)
            @trace(mvnormal(mu, cov), :amp)  

        end
    end
    
    return length(old_gp_timings), length(new_gp_timings)
    
end

function duration_involution(trace, fwd_choices, fwd_ret, proposal_args)
                            
    if fwd_ret == "abort"
        return trace, fwd_choices, 0
    end
                            
    #we need to specify how to go backwards
    #and how to construct the new trace
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    max_elements = source_params["n_elements"]["val"]
    bwd_choices = choicemap()
    new_choices = choicemap()
    element_idx = proposal_args[1]; source_id = proposal_args[2]
    old_k = fwd_ret[1]; new_k = fwd_ret[2]
    
    old_source_choices = get_submap(get_choices(trace),:source => source_id); 
    old_element_choices = get_submap(get_choices(trace),:source => source_id => (:element, element_idx)); 
                                        
    #Set new timing values for (:element, element_idx)
    #Duration
    new_choices[:source => source_id => (:element, element_idx) => :dur_minus_min] = fwd_choices[:duration] - steps["min"]
    if element_idx < old_source_choices[:n_elements]
        #if duration_diff > zero, that means idx'd tone is longer. meaning next tone should have less wait
        duration_diff = (fwd_choices[:duration] - steps["min"]) - old_element_choices[:dur_minus_min]
        new_choices[:source => source_id => (:element, element_idx + 1) => :wait] = old_source_choices[(:element, element_idx + 1) => :wait] - duration_diff
    end
    bwd_choices[:duration] = old_element_choices[:dur_minus_min] + steps["min"]
    

    source_type = source_params["types"][trace[:source => source_id => :source_type]]
    gp_types = source_type == "tone" || source_type == "harmonic" ? [:erb,:amp] : [:amp]
    if new_k > old_k
        for gp_type = gp_types
            new_gp = vcat(old_element_choices[gp_type],fwd_choices[gp_type])
            new_choices[:source => source_id => (:element, element_idx) => gp_type] = new_gp
        end  
    elseif new_k < old_k
        for gp_type = gp_types
            old_gp = old_element_choices[gp_type]
            lenf = (gp_type == :erb || (gp_type == :amp && source_type == "tone")) ? 1 : length(get_element_gp_freqs(audio_sr, steps))  

            bwd_choices[gp_type] = old_gp[((lenf*new_k)+1):end]
            new_choices[:source => source_id =>(:element, element_idx) => gp_type] = old_gp[1:lenf*new_k]
        end                                                          
    end
    
    new_trace, weight = update(trace, get_args(trace), (), new_choices)
    return new_trace, bwd_choices, weight

end


## CHANGE SOME or ALL GP ELEMENTS ONLY
@gen function gp_randomness(trace, element_idx, update_idxs, gp_type, source_id)
    
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    max_elements = source_params["n_elements"]["val"]
    old_choices = get_submap(get_choices(trace),:source => source_id)
    
    old_abs_timings = absolute_timing(old_choices, steps["min"])
    source_type = source_params["types"][old_choices[:source_type]]
    if gp_type == :erb || (gp_type == :amp && source_type == "tone")  #tone and harmonic erb, tone amp
        x = get_element_gp_times(old_abs_timings[element_idx], steps["t"])
    else #harmonic amp, noise amp
        x, _, _ = get_gp_spectrotemporal(old_abs_timings[element_idx], steps, audio_sr)  
    end
    
    # Get which indexes to keep constant and condition on
    old_vals = old_choices[(:element, element_idx)=>gp_type]
    full_idx_set = 1:length(old_vals);
    non_update_idx = [i for i in full_idx_set if ~in(i,update_idxs)]
        
    # Get old GP params to condition on     
    y = old_vals[non_update_idx]
    t_y = x[non_update_idx]
                
    gp_latents = get_submap(get_choices(trace), :source => source_id => gp_type)
    mu, cov = get_cond_mu_cov(x[update_idxs], x[non_update_idx], old_vals[non_update_idx], gp_latents)
    @trace(mvnormal(mu, cov), gp_type)        
                
end

function gp_involution(trace, fwd_choices, fwd_ret, proposal_args)
    
    #we need to specify how to go backwards
    #and how to construct the new trace
    bwd_choices = choicemap()
    new_choices = choicemap()
    element_idx = proposal_args[1] #index of the element we're changing 
    update_idxs = proposal_args[2] #e.g., [3, 4, 5] should be contiguous indexes
    gp_type = proposal_args[3] #:erb or :amp
    source_id = proposal_args[4]
                            
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    max_elements = source_params["n_elements"]["val"]
    old_choices = get_submap(get_choices(trace),:source => source_id)
    old_vals = old_choices[(:element, element_idx)=>gp_type]

    full_idx_set = 1:length(old_vals);
    before_update_idx = [i for i in full_idx_set if (~in(i,update_idxs) && i < minimum(update_idxs))]
    after_update_idx = [i for i in full_idx_set if (~in(i,update_idxs) && i > maximum(update_idxs))]
                
    bwd_choices[gp_type] = old_vals[update_idxs]
    updated_vals = vcat(old_vals[before_update_idx], fwd_choices[gp_type], old_vals[after_update_idx])
    new_choices[:source => source_id => (:element, element_idx) => gp_type] = updated_vals
    
    new_trace, weight = update(trace, get_args(trace), (), new_choices)
    return new_trace, bwd_choices, weight

end            

## SPLIT A Element OR MERGE TWO ADJACENT Elements
@gen function sm_randomness(trace, source_id)
    
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    max_elements = source_params["n_elements"]["val"]
    old_choices = get_submap(get_choices(trace),:source => source_id)
    source_type = source_params["types"][old_choices[:source_type]]
    
    # number of elements in this source
    old_n_elements = old_choices[:n_elements]
    old_abs_timings = absolute_timing(old_choices, steps["min"])
    
    # pick the element from this source to split (or merge with the element after it)
    element_idx = @trace(uniform_discrete(1,old_n_elements),:element_idx)
    element_duration = old_choices[(:element, element_idx) => :dur_minus_min] + steps["min"]
    
    if element_idx == old_n_elements
        # this is the last element in the source, so we cannot merge it with the next element
        # the duration needs to be long enough to be split
        if element_duration > 2*steps["min"]
            p_split = 1
        else
            #if we can't merge and we can't split, abort
            return "abort"
        end
    else
        #if the element is long enough, then we can either split or merge
        p_split = element_duration > 2*steps["min"] ? 0.5 : 0
    end    
    
    #choose to split or not
    is_split = @trace(bernoulli(p_split),:is_split)
    new_n_elements = old_n_elements + (is_split ? 1 : -1)
    
    if is_split
        
        #We need to sample the duration of the first element
        #and the wait of the second element
        #In the involution we will adjust the duration of the second element
        #so that it keeps its old offset 
        
        old_onset = old_abs_timings[element_idx][1]; old_offset = old_abs_timings[element_idx][2]
        old_duration = old_offset - old_onset
        ## Need to direct the choice towards the place in the element that is likely to need splitting
        # Check whether there are points in the element that have big derivatives 
        # Weight the probability distribution by the magnitude of the derivative
        
        if source_type == "tone" || source_type == "harmonic" #tone                                    
            feature = old_choices[(:element, element_idx) => :erb]
        else #noise
            tfs, ts, fs = get_gp_spectrotemporal([old_onset, old_offset], steps, audio_sr)
            amp = old_choices[(:element, element_idx) => :amp]
            feature = mean(transpose(reshape(amp, (length(fs), length(ts))));dims=2);
        end
        old_t = get_element_gp_times([old_onset, old_offset], steps["t"]) .- old_onset
        old_t = old_t[2:end]
        dfdt = abs.(feature[2:end] .- feature[1:end-1])./steps["t"]; 
        dfdt = [dfdt[i] + 1e-5 for i in 1:length(dfdt) if steps["min"] < old_t[i] < (old_duration-steps["min"])-0.001]
        old_t = [t for t in old_t if steps["min"] < t < (old_duration-steps["min"])-0.001] 
        
        #new_duration = @trace(uniform(steps["min"], old_duration-steps["min"]), :duration)
        if length(dfdt) > 0 
            bounds = vcat([steps["min"]], old_t, [old_duration-steps["min"]])
            append!(dfdt, dfdt[end])                                                            
            probs = dfdt./sum(dfdt)
            new_duration = @trace(piecewise_uniform(bounds, probs), :duration)
        else
            new_duration = @trace(uniform(steps["min"], old_duration-steps["min"]),:duration)                                                                  
        end
        #The new wait should be heavily skewed towards being soon after the new duration                                     
#        new_wait = @trace(uniform(0, old_duration-new_duration-steps["min"]), :wait)
        max_wait = old_duration-new_duration-steps["min"];
        bounds = collect(range(0, stop=max_wait, length=max(2,Int(floor(max_wait/steps["t"])))))
        probs = Float32.(collect(((length(bounds) - 1):-1:1)))
        probs ./= sum(probs)
        new_wait = @trace(piecewise_uniform(bounds, probs), :wait)
        
    else
        
        old_t_1 = get_element_gp_times(old_abs_timings[element_idx], steps["t"])
        old_t_2 = get_element_gp_times(old_abs_timings[element_idx + 1], steps["t"])
        
        new_onset = old_abs_timings[element_idx][1]
        new_offset = old_abs_timings[element_idx + 1][2]
        new_t = get_element_gp_times([new_onset, new_offset], steps["t"])                                                           
                                                                    
        old_t = append!(old_t_1, old_t_2)                                                           
        add_t = [i for i in new_t if ~in(i,old_t)]
        gp_types = source_type == "tone" || source_type == "harmonic" ? [:erb, :amp] : [:amp]
                                                                                
        if length(add_t) > 0 
            for gp_type in gp_types
                gp_latents = get_submap(get_choices(trace), :source => source_id => gp_type)
                old_x = (gp_type == :erb || (gp_type == :amp && source_type == "tone")) ? old_t : cross_tf(old_t, get_element_gp_freqs(audio_sr, steps))  
                add_x = (gp_type == :erb || (gp_type == :amp && source_type == "tone")) ? add_t : cross_tf(add_t, get_element_gp_freqs(audio_sr, steps))  
                old_gp = vcat(old_choices[(:element,element_idx) => gp_type], old_choices[(:element,element_idx+1) => gp_type])
                mu, cov = get_cond_mu_cov(add_x, old_x, old_gp, gp_latents)
                @trace(mvnormal(mu, cov), gp_type) 
            end                         
        end
        
    end      
                
end
                                                   
function sm_involution(trace, fwd_choices, fwd_ret, proposal_args)
    
    if fwd_ret == "abort"
        return trace, fwd_choices, 0
    end
                                                        
    #we need to specify how to go backwards
    #and how to construct the new trace
    bwd_choices = choicemap()
    new_choices = choicemap()
    source_id = proposal_args[1]
    
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    max_elements = source_params["n_elements"]["val"]
    old_choices = get_submap(get_choices(trace),:source => source_id)
    source_type = source_params["types"][old_choices[:source_type]]
    gp_types = source_type == "tone" || source_type == "harmonic" ? [:erb, :amp] : [:amp] 
    
    old_n_elements = old_choices[:n_elements]
    old_abs_timings = absolute_timing(old_choices, steps["min"])
    
    is_split = fwd_choices[:is_split]
    bwd_choices[:is_split] = ~is_split
    new_n_elements = old_n_elements + (is_split ? 1 : -1)
    new_choices[:source => source_id => :n_elements] = new_n_elements
                    
    element_idx = fwd_choices[:element_idx]
    bwd_choices[:element_idx] = element_idx
                                                                            
    if is_split
        
        #element element_idx (first of two split-elements)
        new_choices[:source => source_id => (:element, element_idx) => :dur_minus_min] = fwd_choices[:duration] - steps["min"]
        split_onset_1 = old_abs_timings[element_idx][1]; split_offset_1 = split_onset_1 + fwd_choices[:duration]
        new_t_1 = get_element_gp_times([split_onset_1, split_offset_1], steps["t"])
        new_k_1 = length(new_t_1)                                                                                  
        for gp_type = gp_types
            lenf = (gp_type == :erb || (gp_type == :amp && source_type == "tone")) ? 1 : length(get_element_gp_freqs(audio_sr, steps))                                                                             
            new_choices[:source => source_id => (:element, element_idx) => gp_type] = old_choices[(:element,element_idx) => gp_type][1:lenf*new_k_1] 
        end
        
        #element element_idx + 1 (second of two split-elements)
        new_choices[:source => source_id => (:element, element_idx + 1) => :wait] = fwd_choices[:wait]
        split_onset_2 = split_offset_1 + fwd_choices[:wait]; split_offset_2 = old_abs_timings[element_idx][2]; 
        split_duration_2 = split_offset_2 - split_onset_2;
        new_choices[:source => source_id => (:element, element_idx + 1) => :dur_minus_min ] = split_duration_2 - steps["min"] 
        new_t_2 = get_element_gp_times([split_onset_2, split_offset_2], steps["t"])
        new_k_2 = length(new_t_2)
        for gp_type = gp_types
            lenf = (gp_type == :erb || (gp_type == :amp && source_type == "tone")) ? 1 : length(get_element_gp_freqs(audio_sr, steps))                                                                             
            new_choices[:source => source_id => (:element, element_idx + 1) => gp_type] = old_choices[(:element,element_idx) => gp_type][end-(lenf*new_k_2)+1:end]
        end                
    
        #shift all elements following the split-elements
        for i = (element_idx + 2) : new_n_elements
            set_submap!(new_choices, :source => source_id => (:element, i), get_submap(old_choices, (:element,i-1)))                 
        end
        
        #define backwards choices
        if source_type == "tone" #tone
            new_t = append!(new_t_1, new_t_2)
            old_t = get_element_gp_times(old_abs_timings[element_idx], steps["t"])                                                                          
            out_t = [i for i in 1:length(old_t) if ~in(old_t[i],new_t)]
            out_t = Dict(:erb=>out_t, :amp=>out_t)
        elseif source_type == "noise" #noise
            new_tf_1, _, _ = get_gp_spectrotemporal([split_onset_1, split_offset_1], steps, audio_sr)
            new_tf_2, _, _ = get_gp_spectrotemporal([split_onset_2, split_offset_2], steps, audio_sr)
            new_tf = append!(new_tf_1,new_tf_2)
            old_tf,_,_ = get_gp_spectrotemporal(old_abs_timings[element_idx], steps, audio_sr)
            out_t = [i for i in 1:length(old_tf) if ~in(old_tf[i],new_tf)]
            out_t = Dict(:amp=>out_t)                                                                                            
        elseif source_type == "harmonic"

            #fundamental frequency
            new_t = append!(new_t_1, new_t_2)
            old_t = get_element_gp_times(old_abs_timings[element_idx], steps["t"])                                                                          
            erb_out_t = [i for i in 1:length(old_t) if ~in(old_t[i],new_t)]

            #filter
            new_tf_1, _, _ = get_gp_spectrotemporal([split_onset_1, split_offset_1], steps, audio_sr)
            new_tf_2, _, _ = get_gp_spectrotemporal([split_onset_2, split_offset_2], steps, audio_sr)
            new_tf = append!(new_tf_1,new_tf_2)
            old_tf,_,_ = get_gp_spectrotemporal(old_abs_timings[element_idx], steps, audio_sr)
            amp_out_t = [i for i in 1:length(old_tf) if ~in(old_tf[i],new_tf)]
            
            out_t = Dict(:erb=>erb_out_t, :amp=>amp_out_t)  

        end        
        for gp_type in gp_types
            if length(out_t[gp_type]) > 0
                bwd_choices[gp_type] = old_choices[(:element, element_idx) => gp_type][out_t[gp_type]]
            end
        end
                                                            
    else
        
        #merge element_idx and element_idx+1
        new_choices[:source => source_id => (:element, element_idx) => :wait] = old_choices[(:element, element_idx) => :wait]
        new_onset = old_abs_timings[element_idx][1]; new_offset = old_abs_timings[element_idx + 1][2]; new_duration = new_offset - new_onset;               
        new_choices[:source => source_id => (:element, element_idx) => :dur_minus_min] = new_duration - steps["min"]
        for gp_type in gp_types
            old_gp_1 = old_choices[(:element, element_idx) => gp_type]
            old_gp_2 = old_choices[(:element, element_idx + 1) => gp_type]
            connecting_gp = has_value(fwd_choices, gp_type) ? fwd_choices[gp_type] : []
            new_choices[:source => source_id => (:element, element_idx) => gp_type] = vcat(old_gp_1, connecting_gp, old_gp_2)
        end
        #shift all following elements
        for i = (element_idx + 1) : new_n_elements
            set_submap!(new_choices, :source => source_id => (:element, i), get_submap(old_choices, (:element, i + 1)))
        end
        #define backwards choices                          
        bwd_choices[:duration] = old_choices[(:element, element_idx) => :dur_minus_min] + steps["min"]
        bwd_choices[:wait] = old_choices[(:element, element_idx + 1) => :wait]
    
    end

    new_trace, weight = update(trace, get_args(trace), (), new_choices)
    return new_trace, bwd_choices, weight

end

## SWITCH AN ELEMENT FROM ONE STREAM TO ANOTHER

# @gen function rewrite_switch_randomness(trace)
    
#     source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
#     max_elements = source_params["n_elements"]["val"]
#     old_choices = get_choices(trace)
#     #onset/offset information for each element in each source: all source timings
#     all_source_timings = []
#     old_n_sources = old_choices[:n_sources]
#     for i = 1:old_n_sources
#         #list of lists of times
#         #[ (element 1)[onset, offset], (element2)[onset, offset], ... ]
#         old_abs_timings = absolute_timing(get_submap(old_choices, :source => i), steps["min"])
#         push!(all_source_timings,old_abs_timings) 
#     end
    
    
#     origin = @trace(uniform_discrete(1, old_n_sources), :origin)
#     old_n_elements = old_choices[:source => origin => :n_elements]
#     element_idx = @trace(uniform_discrete(1,old_n_elements),:element_idx)
#     onset = all_source_timings[origin][element_idx][1];
#     offset = all_source_timings[origin][element_idx][2];
    
#     #Find the sources into which a element can be switched 
#     source_switch = []; which_spot = [];
#     for i = 1:old_choices[:n_sources]
#         # Only can switch things between streams of the same source_type. may not want this to be true.                                                                                                            
#         if i == origin || (old_choices[:source => i => :source_type] != old_choices[:source => origin => :source_type])
#             append!(source_switch, 0); append!(which_spot, 0)
#         else
#             source_nt = old_choices[:source => i => :n_elements]
#             timings = all_source_timings[i];
#             for j = 1:source_nt + 1

#                 #switch into spot before the first element
#                 if j == 1 
#                     fits = (0 < onset) && (offset < timings[j][1])
#                 elseif 1 < j <= source_nt
#                     fits = (timings[j-1][2] < onset) && (offset < timings[j][1])
#                 elseif j == source_nt + 1
#                     fits = (timings[j - 1][2] < onset)
#                 end

#                 if fits
#                     append!(source_switch, 1)
#                     append!(which_spot, j)
#                     break
#                 elseif j == source_nt + 1
#                     append!(source_switch, 0)
#                     append!(which_spot, 0)
#                 end

#             end
#         end
#     end
    
#     #Decide whether to move the element into an existing source
#     #Or make a new source, where it will be the only element
#     switch_to_existing_source = sum(source_switch)
#     switch_to_new_source = (old_n_elements > 1 && old_n_sources < source_params["n_sources"]["val"]) ? 1 : 0 ## currently hard coded that we're using a uniform distribution
#     switch_weights = [switch_to_existing_source, switch_to_new_source]
#     if sum(switch_weights) == 0
#         return "abort"
#     end
#     ps = switch_weights./sum(switch_weights)
#     new_source = @trace(bernoulli(ps[2]), :new_source)
#     #Decide the idx of the destination source
#     #If it's a new source, it can go before any of the old sources or at the end
#     #If it's an old source, you need to choose from the ones in source_switch
#     destination_ps = new_source ? fill(1/(old_choices[:n_sources] + 1), old_choices[:n_sources] + 1) : source_switch./sum(source_switch)
#     destination = @trace(categorical(destination_ps), :destination)
    
#     ## change the source level variables to increase probabilities of acceptance
#     source_type = source_params["types"][old_choices[:source => origin => :source_type]] #because we retain the source_type for switches
#     tp_types = keys(source_params["tp"])
#     gp_types = source_type == "tone" || source_type == "harmonic" ? ["erb", "amp"] : ["amp"]
#     source_vars = cat(tp_types, gp_types, dims=1)

#     #tp: wait & dur_minus_min --> a (variability), mu (mean)
#     #gp: erb --> mu, scale, sigma, noise
#     #    amp --> mu, scale, sigma, noise OR mu, scale_t, scale_f, sigma, noise

#     dest_elements_list = Dict()
#     for source_var in source_vars 
#         #Collecting the description of all the elements in the destination stream
#         source_var_sym = Symbol(source_var)
#         dest_elements_list[source_var] = []
#         if ~new_source
#             for j = 1:old_choices[:source => destination => :n_elements]
#                 compilefunc = source_var == "erb" || source_var == "amp" ? append! : push! 
#                 compilefunc(dest_elements_list[source_var], old_choices[:source => destination => (:element, j) => source_var_sym])
#                 end
#             end
#         end
#         append!(dest_elements_list[source_var], old_choices[:source => origin => (:element, element_idx) => source_var_sym])
#     end

#     #For temporal parameters, use method of moments to find fit to gamma distributions 
#     for source_var in tp_types
#         m = mean(dest_elements_list[source_var])
#         s = std(dest_elements_list[source_var])
#         for sv in keys(source_params["tp"][source_var])            
#             est = sv == "mu" ? m : (m^2)/(s^2)
#             a = ?
#             b = a/est
#             @trace(gamma(a,b), :dest => Symbol(source_var) => Symbol(sv))
#         end
#     end

#     for gp 


#     gp_types =  source_type == "tone" || source_type == "harmonic" ? [:erb, :amp] : [:amp]
#     source_attributes = Dict(:erb=>[:scale, :sigma, :epsilon], :amp=> (source_type == "tone") ?  [:scale, :sigma, :epsilon]  : [:scale_t, :scale_f, :sigma, :epsilon])                                                                                                        
#     for gp_type = gp_types 
#         dest_source_mus[gp_type] = []
#         if ~new_source                                                                                            
#             for j = 1:old_choices[:source => destination => :n_elements]
#                 append!(dest_source_mus[gp_type], old_choices[:source => destination => (:element, j) => gp_type])
#             end
#         end
#         append!(dest_source_mus[gp_type], old_choices[:source => origin => (:element, element_idx) => gp_type])                                                                                         
#         mu = mean(dest_source_mus[gp_type]) 
#         sigma = std(dest_source_mus[gp_type])  #1#gp_type[1] == :erb ? 1 : 0.1                                                                                            
#         dest_source_mus[gp_type] = @trace(normal(mu, sigma > 1 ? sigma : 1), :dest => gp_type => :mu) 
#         #Also sample all of the other 
#         for a in source_attributes[gp_type]
#             @trace(gamma(1,1), :dest => gp_type => a) ##TO DO: THIS COULD BE MORE SPECIFIC TO THE PRIORS
#         end
#     end                                                                                                                 
                                                                                                                    
#     if old_choices[:source => origin => :n_elements] > 1  
#         #there must be an already existing origin source
#         origin_source_mus = Dict()
#         for gp_type = gp_types
#             origin_source_mus[gp_type] = []  
#             for j = [jj for jj in 1:old_choices[:source => origin => :n_elements] if jj != element_idx]
#                 append!(origin_source_mus[gp_type], old_choices[:source => origin => (:element, j) => gp_type])
#             end
#             mu = mean(origin_source_mus[gp_type])  
#             sigma = std(origin_source_mus[gp_type]) #1#gp_type[1] == :erb ? 1 : 0.1                                                                                                          
#             origin_source_mus[gp_type] = @trace(normal(mu,  sigma > 1 ? sigma : 1), :orig => gp_type => :mu )
#             for a in source_attributes[gp_type]
#                 @trace(gamma(1,1), :orig => gp_type => a) ##TO DO: THIS COULD BE MORE SPECIFIC TO THE PRIORS
#             end
#         end   
#     end 
                                                                                                                                                                                                                                      
#     return which_spot, all_source_timings
               
# end
                                                                                                                                    
@gen function switch_randomness(trace)
    
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    max_elements = source_params["n_elements"]["val"]
    old_choices = get_choices(trace)
    #onset/offset information for each element in each source: all source timings
    all_source_timings = []
    old_n_sources = old_choices[:n_sources]
    for i = 1:old_n_sources
        #list of lists of times
        #[ (element 1)[onset, offset], (element2)[onset, offset], ... ]
        old_abs_timings = absolute_timing(get_submap(old_choices, :source => i), steps["min"])
        push!(all_source_timings,old_abs_timings) 
    end
    
    
    origin = @trace(uniform_discrete(1, old_n_sources), :origin)
    old_n_elements = old_choices[:source => origin => :n_elements]
    element_idx = @trace(uniform_discrete(1,old_n_elements),:element_idx)
    onset = all_source_timings[origin][element_idx][1];
    offset = all_source_timings[origin][element_idx][2];
    
    #Find the sources into which a element can be switched 
    source_switch = []; which_spot = [];
    for i = 1:old_choices[:n_sources]
        # Only can switch things between streams of the same source_type. may not want this to be true.                                                                                                            
        if i == origin || (old_choices[:source => i => :source_type] != old_choices[:source => origin => :source_type])
            append!(source_switch, 0); append!(which_spot, 0)
        else
            source_nt = old_choices[:source => i => :n_elements]
            timings = all_source_timings[i];
            for j = 1:source_nt + 1

                #switch into spot before the first element
                if j == 1 
                    fits = (0 < onset) && (offset < timings[j][1])
                elseif 1 < j <= source_nt
                    fits = (timings[j-1][2] < onset) && (offset < timings[j][1])
                elseif j == source_nt + 1
                    fits = (timings[j - 1][2] < onset)
                end

                if fits
                    append!(source_switch, 1)
                    append!(which_spot, j)
                    break
                elseif j == source_nt + 1
                    append!(source_switch, 0)
                    append!(which_spot, 0)
                end

            end
        end
    end
    
    #Decide whether to move the element into an existing source
    #Or make a new source, where it will be the only element
    switch_to_existing_source = sum(source_switch)
    switch_to_new_source = (old_n_elements > 1 && old_n_sources < 10) ? 1 : 0 ## currently hard coded: 10 sources
    switch_weights = [switch_to_existing_source, switch_to_new_source]
    if sum(switch_weights) == 0
        return "abort"
    end
    ps = switch_weights./sum(switch_weights)
    new_source = @trace(bernoulli(ps[2]), :new_source)
    #Decide the idx of the destination source
    #If it's a new source, it can go before any of the old sources or at the end
    #If it's an old source, you need to choose from the ones in source_switch
    destination_ps = new_source ? fill(1/(old_choices[:n_sources] + 1), old_choices[:n_sources] + 1) : source_switch./sum(source_switch)
    destination = @trace(categorical(destination_ps), :destination)
    
    ## change the means so that there's the best possibility for acceptance
    dest_source_mus = Dict()
    source_type = source_params["types"][old_choices[:source => origin => :source_type]]
    gp_types =  source_type == "tone" || source_type == "harmonic" ? [:erb, :amp] : [:amp]
    source_attributes = Dict(:erb=>[:scale, :sigma, :epsilon], :amp=> (source_type == "tone") ?  [:scale, :sigma, :epsilon]  : [:scale_t, :scale_f, :sigma, :epsilon])                                                                                                        
    for gp_type = gp_types 
        dest_source_mus[gp_type] = []
        if ~new_source                                                                                            
            for j = 1:old_choices[:source => destination => :n_elements]
                append!(dest_source_mus[gp_type], old_choices[:source => destination => (:element, j) => gp_type])
            end
        end
        append!(dest_source_mus[gp_type], old_choices[:source => origin => (:element, element_idx) => gp_type])                                                                                         
        mu = mean(dest_source_mus[gp_type]) 
        sigma = std(dest_source_mus[gp_type])  #1#gp_type[1] == :erb ? 1 : 0.1                                                                                            
        dest_source_mus[gp_type] = @trace(normal(mu, sigma > 1 ? sigma : 1), :dest => gp_type => :mu) 
        #Also sample all of the other 
        for a in source_attributes[gp_type]
            @trace(gamma(1,1), :dest => gp_type => a) ##TO DO: THIS COULD BE MORE SPECIFIC TO THE PRIORS
        end
    end                                                                                                                 
                                                                                                                    
    if old_choices[:source => origin => :n_elements] > 1  
        #there must be an already existing origin source
        origin_source_mus = Dict()
        for gp_type = gp_types
            origin_source_mus[gp_type] = []  
            for j = [jj for jj in 1:old_choices[:source => origin => :n_elements] if jj != element_idx]
                append!(origin_source_mus[gp_type], old_choices[:source => origin => (:element, j) => gp_type])
            end
            mu = mean(origin_source_mus[gp_type])  
            sigma = std(origin_source_mus[gp_type]) #1#gp_type[1] == :erb ? 1 : 0.1                                                                                                          
            origin_source_mus[gp_type] = @trace(normal(mu,  sigma > 1 ? sigma : 1), :orig => gp_type => :mu )
            for a in source_attributes[gp_type]
                @trace(gamma(1,1), :orig => gp_type => a) ##TO DO: THIS COULD BE MORE SPECIFIC TO THE PRIORS
            end
        end   
    end 
                                                                                                                                                                                                                                      
    return which_spot, all_source_timings
               
end                                                                                                                                    
                                                    
function switch_involution(trace, fwd_choices, fwd_ret, proposal_args)
    
    if fwd_ret == "abort"
        return trace, fwd_choices, 0
    end
    #we need to specify how to go backwards
    #and how to construct the new trace
    bwd_choices = choicemap()
    new_choices = choicemap()
    which_gaps = fwd_ret[1]; all_source_timings = fwd_ret[2];
    
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    max_elements = source_params["n_elements"]["val"]
    old_choices = get_choices(trace)
    old_n_sources = old_choices[:n_sources]; 
    
    ## indexes for moving a element from origin to destination source
    origin_idx = fwd_choices[:origin]
    new_source = fwd_choices[:new_source]
    destination_idx = fwd_choices[:destination]
    element_switch_idx = fwd_choices[:element_idx]
                                                                                                                             
    
    old_origin_nt = old_choices[:source => origin_idx => :n_elements]
    old_destination_nt = new_source ? 0 : old_choices[:source => destination_idx => :n_elements]
    which_gap = new_source ? 1 : which_gaps[destination_idx]
    source_type = source_params["types"][old_choices[:source => origin_idx => :source_type]]
    gp_types = source_type == "tone" || source_type == "harmonic" ? [:erb, :amp] : [:amp]
    element_attributes = append!([:wait, :dur_minus_min], gp_types)
    element_attributes_no_wait = append!([:dur_minus_min], gp_types)
    source_attributes = Dict(:erb=>[:mu, :scale, :sigma, :epsilon], :amp=> (source_type == "tone") ?  [:mu,:scale, :sigma, :epsilon]  : [:mu,:scale_t, :scale_f, :sigma, :epsilon])  

    ##Get all the properties of the switch element in the new source
    switch_element = Dict()
    #absolute onset and offset stay the same, so do duration and gps
    switch_element[:onset] = all_source_timings[origin_idx][element_switch_idx][1] 
    switch_element[:offset] = all_source_timings[origin_idx][element_switch_idx][2]         
    switch_element[:dur_minus_min]= old_choices[:source => origin_idx => (:element, element_switch_idx) => :dur_minus_min]
    for gp_type = gp_types
        switch_element[gp_type]=old_choices[:source => origin_idx => (:element, element_switch_idx) =>gp_type]
    end 
    #wait depends on what is before the switch_element in the destination stream
    prev_offset = (which_gap == 1) ? 0 : all_source_timings[destination_idx][which_gap - 1][2] 
    switch_element[:wait] = switch_element[:onset] - prev_offset;

    ##compute new WAITS OF elementS FOLLOWING SWITCH element, in both destination and origin
    #inserting switch_element before a element in destination source
    if which_gap <= old_destination_nt
        dest_after_wait = all_source_timings[destination_idx][which_gap][1] - switch_element[:offset]
    end
    #removing switch_element before a element in origin source
    if element_switch_idx < old_origin_nt
        prev_offset = element_switch_idx == 1 ? 0 : all_source_timings[origin_idx][element_switch_idx - 1][2]
        orig_after_wait = all_source_timings[origin_idx][element_switch_idx + 1][1] - prev_offset
    end
    
    if old_origin_nt == 1 
        # If the origin stream had only one element in it, it should be removed
        # The switch_element will not be moved into a new stream,
        # so n_sources should always decrease by 1
        
        # new_source = false
        # destination_idx chooses an existing source 
        
        # if destination_idx is larger than origin_idx
        # the idx of the destination source needs to be shifted down one
        # and any sources after the destination source need to be shifted down one
        
        # if destination_idx is smaller than origin_idx
        # the idx of the destination source can remain the same, 
        # but others may be shifted down one 

        new_choices[:n_sources] = old_n_sources - 1
        #Get indexes of old sources that must be changed 
        new_idx = 1:(old_n_sources - 1)
        #the OLD labels after the origin index are shifted up one because their new labels will be one smaller
        old_idx = [(n >= origin_idx ? (n + 1) : n) for n in new_idx] 
        #Only need to change old indexes that are greater than or equal to the origin idx
        #Get rid of (old_idx < origin_idx) because those won't change...
        #...as well as the destination index, which will be treated on its own
        matching_new_idx = [new_idx[i] for i in 1:length(new_idx) if (old_idx[i] != destination_idx)]
        old_idx = [o for o in old_idx if (o != destination_idx)]
        
        ##Shift sources that do not change
        for i = 1:length(old_idx)                        
            set_submap!(new_choices, :source=>matching_new_idx[i], get_submap(old_choices,:source=>old_idx[i]))                        
        end
        
        ##Deal with destination source specifically
        old_destination_idx = destination_idx
        new_destination_idx = old_destination_idx > origin_idx ? old_destination_idx - 1 : old_destination_idx 
        old_nt = old_choices[:source=>old_destination_idx=>:n_elements]

        #Get source attributes
        new_choices[:source => new_destination_idx => :n_elements] = old_nt + 1
        new_choices[:source => new_destination_idx => :source_type] = old_choices[:source=>old_destination_idx=>:source_type]
        for gp_type in gp_types
            for a in source_attributes[gp_type]
                new_choices[:source => new_destination_idx => gp_type => a] = fwd_choices[:dest => gp_type => a]
            end
        end
        
        #Switch element
        for a in element_attributes
            new_choices[:source => new_destination_idx => (:element, which_gap) => a] = switch_element[a]
        end
                                
        #All elements before the switch_element stay the same
        if which_gap > 1
            for j = 1:which_gap - 1
                set_submap!(new_choices, :source => new_destination_idx => (:element, j), get_submap(old_choices,:source=>old_destination_idx=>(:element,j)))
            end
        end
        #If there are any elements after the switch_element they must be increased in index by one
        if which_gap <= old_nt #comes before one of the old elements
            for new_element_idx = (which_gap + 1):(old_nt + 1)
                new_choices[:source => new_destination_idx => (:element, new_element_idx) => :wait] = (new_element_idx == (which_gap + 1)) ? dest_after_wait : old_choices[:source => old_destination_idx => (:element,new_element_idx-1) => :wait]
                for a in element_attributes_no_wait
                    new_choices[:source => new_destination_idx => (:element, new_element_idx) => a] = old_choices[:source=>old_destination_idx=>(:element,new_element_idx-1)=>a]
                end 
            end
        end

        bwd_choices[:origin] = new_destination_idx
        bwd_choices[:new_source] = true
        for gp_type = gp_types
            for a in source_attributes[gp_type]
                bwd_choices[:dest => gp_type => a] = old_choices[:source => origin_idx => gp_type => a]
                bwd_choices[:orig => gp_type => a] = old_choices[:source => destination_idx => gp_type => a]
            end
        end
        bwd_choices[:destination] = origin_idx
        bwd_choices[:element_idx] = which_gap
            
    elseif new_source
        # we put the element in a new stream 
        # we keep the origin stream as well
        # so n_sources increases by 1
        # need to shift all the sources after the destination_idx
                                
        new_choices[:n_sources] = old_n_sources + 1
        new_choices[:source => destination_idx => :source_type] = old_choices[:source => origin_idx => :source_type]
        
        ##Create new destination source with a single element in it
        new_choices[:source => destination_idx => :n_elements] = 1
        for gp_type in gp_types
            for a in source_attributes[gp_type]
                new_choices[:source => destination_idx => gp_type => a] = fwd_choices[:dest => gp_type => a]
            end
        end
        for a in element_attributes
            new_choices[:source => destination_idx => (:element, 1) => a] = switch_element[a]
        end

        ##in origin source, move all elements down one index if they're after the switch index
        old_origin_idx = origin_idx
        new_origin_idx = origin_idx >= destination_idx ? origin_idx + 1 : origin_idx
        old_nt = old_choices[:source => origin_idx => :n_elements]
        new_choices[:source => new_origin_idx => :n_elements] = old_nt - 1
        new_choices[:source => new_origin_idx => :source_type] = old_choices[:source => origin_idx => :source_type]
        for gp_type in gp_types
            for a in source_attributes[gp_type]
                new_choices[:source => new_origin_idx => gp_type => a] = fwd_choices[:orig => gp_type => a]
            end
        end
        if element_switch_idx > 1
            for j = 1:element_switch_idx - 1
                set_submap!(new_choices, :source => new_origin_idx => (:element, j), get_submap(old_choices,:source=>old_origin_idx=>(:element,j)))
            end
        end                        
        if element_switch_idx < old_nt
            for old_element_idx = (element_switch_idx + 1):old_nt
                new_choices[:source => new_origin_idx => (:element, old_element_idx-1)=>:wait] = (old_element_idx == (element_switch_idx + 1)) ? orig_after_wait : old_choices[:source => origin_idx => (:element, old_element_idx)=> :wait]
                for a in element_attributes_no_wait
                    new_choices[:source => new_origin_idx => (:element, old_element_idx-1) => a] = old_choices[:source=>origin_idx=>(:element,old_element_idx)=>a]
                end 
            end
        end
                                
        ##shift all sources after destination_idx up one
        if destination_idx < new_choices[:n_sources]
            shift_idxs = [i for i in (destination_idx+1):new_choices[:n_sources] if i != new_origin_idx]
            for i in shift_idxs
                set_submap!(new_choices, :source=>i, get_submap(old_choices, :source=>i-1))
            end
        end
                                
        bwd_choices[:origin] = destination_idx
        bwd_choices[:new_source] = false  
        for gp_type = gp_types
            for a in source_attributes[gp_type]
                bwd_choices[:dest => gp_type => a] = old_choices[:source => origin_idx => gp_type => a]
            end
        end
        bwd_choices[:destination] = new_origin_idx
        bwd_choices[:element_idx] = 1
            
    else
        # we put the element in an old stream, and keep the origin stream
        # streams do not have to be shifted 
        # new_source = false
        ##in origin source, move all elements to earlier index if they're after the switch index
        old_nt = old_choices[:source => origin_idx => :n_elements]
        new_choices[:source => origin_idx => :n_elements] = old_nt - 1
        for gp_type = gp_types
            for a in source_attributes[gp_type]
                new_choices[:source => origin_idx => gp_type => a] = fwd_choices[:orig => gp_type => a]
            end
        end
        if element_switch_idx < old_nt
            for old_element_idx = (element_switch_idx + 1):old_nt
                new_choices[:source => origin_idx => (:element, old_element_idx-1)=>:wait] = (old_element_idx == (element_switch_idx + 1)) ? orig_after_wait : old_choices[:source => origin_idx => (:element, old_element_idx)=>:wait]
                for a in element_attributes_no_wait
                    new_choices[:source => origin_idx => (:element, old_element_idx-1) => a] = old_choices[:source=>origin_idx=>(:element,old_element_idx)=>a]
                end 
            end
        end
            
        ##in destination source, insert element and then shift elements to later index
        old_nt = old_choices[:source => destination_idx => :n_elements]
        new_choices[:source => destination_idx => :n_elements] = old_nt + 1
        for gp_type = gp_types
            for a in source_attributes[gp_type]
                new_choices[:source => destination_idx => gp_type => a] = fwd_choices[:dest => gp_type => a]
            end
        end
        #Switch element
        for a in element_attributes
            new_choices[:source => destination_idx => (:element, which_gap) => a] = switch_element[a]
        end
        #elements after switch_element
        if which_gap <= old_nt
            for new_element_idx = (which_gap + 1):(old_nt + 1)
                new_choices[:source => destination_idx =>(:element,new_element_idx)=>:wait] = (new_element_idx == (which_gap + 1)) ? dest_after_wait : old_choices[:source => destination_idx => (:element, new_element_idx-1)=> :wait]
                for a in element_attributes_no_wait
                    new_choices[:source => destination_idx => (:element, new_element_idx) => a] = old_choices[:source=>destination_idx=>(:element,new_element_idx-1)=>a]
                end
            end
        end
            
        bwd_choices[:origin] = destination_idx
        bwd_choices[:destination] = origin_idx
        bwd_choices[:element_idx] = which_gap
        bwd_choices[:new_source] = false
        for gp_type = gp_types
            for a in source_attributes[gp_type]
                bwd_choices[:dest => gp_type => a] = old_choices[:source => origin_idx => gp_type => a]
                bwd_choices[:orig => gp_type => a] = old_choices[:source => destination_idx => gp_type => a]
            end
        end
                                            
              
    end
    new_trace, weight = update(trace, get_args(trace), (), new_choices)
    return new_trace, bwd_choices, weight

end

## SWAP SOURCE ORDERS

@gen function swap_sources_randomness(trace)
    n1 = @trace(uniform_discrete(1, trace[:n_sources]), :to_move)
    return n1
end

function swap_sources_involution(trace, fwd_choices, fwd_ret, proposal_args)
    new_choices = choicemap()
    n1 = fwd_ret
    n_sources = trace[:n_sources]
    set_submap!(new_choices, :source => n1, get_submap(get_choices(trace), :source => n_sources))
    set_submap!(new_choices, :source => n_sources, get_submap(get_choices(trace), :source => n1))
    new_trace, = update(trace, get_args(trace), (), new_choices)
    return new_trace, fwd_choices, 0
end

##Switch source type
@gen function type_randomness(trace, source_id)
    
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    source_types = source_params["types"]
    old_choices = get_submap(get_choices(trace), :source => source_id)

    source_type_idx = old_choices[:source_type]
    old_source_type =  source_types[source_type_idx]    
    new_source_type = (old_source_type == "tone" || old_source_type == "noise") ? "harmonic" : (@trace(bernoulli(0.5), :is_it_a_tone) ? "tone" : "noise") 
    
    new_choices = choicemap(); bwd_choices = choicemap();
    new_source_idx = findall(x -> x==new_source_type, source_types)[1]
    new_choices[:source => source_id => :source_type] = new_source_idx
    if old_source_type == "tone"
        bwd_choices[:is_it_a_tone] = true
    elseif old_source_type == "noise"
        bwd_choices[:is_it_a_tone] = false
    end
    
    
    n_elements = old_choices[:n_elements]
    old_abs_timings = absolute_timing(old_choices, steps["min"])
    new_gp_latents = Dict()
    if old_source_type == "tone" && new_source_type == "harmonic"
        
        new_gp_latents[:amp] = Dict()
        
        #trace[:source => source_id => (:element, i) => :erb] stays the same
        #Need to Sample 2D AMPS
        sp = source_params["gp"]["amp"]["2D"]
        for k in keys(sp)
            sk = Symbol(k)
            new_gp_latents[:amp][sk] = @trace(sp[k]["dist"](sp[k]["args"]...), :amp => sk)
            # new_gp_latents[:amp][:mu] = @trace(normal(15, 10), :amp => :mu) #spectrum level!
            # new_gp_latents[:amp][:scale_t] = @trace(gamma(0.5,1), :amp => :scale_t) 
            # new_gp_latents[:amp][:scale_f] = @trace(gamma(2,1), :amp => :scale_f) 
            # new_gp_latents[:amp][:sigma] = @trace(gamma(10,1), :amp => :sigma) 
            # new_gp_latents[:amp][:epsilon] = @trace(gamma(1,1), :amp => :epsilon) 
        end
        
        for i = 1:n_elements
            tfs, ts, fs = get_gp_spectrotemporal(old_abs_timings[i], steps, audio_sr)
            mu, cov = get_mu_cov(tfs, new_gp_latents[:amp]) #TFS for 2D tone
            new_choices[:source => source_id => (:element, i) => :amp] = @trace(mvnormal(mu, cov), (:element, i) => :amp)
            bwd_choices[(:element, i) => :amp] = old_choices[(:element, i) => :amp]
        end
        
        sp = source_params["gp"]["amp"]["1D"]
        for k in keys(sp)
            sk = Symbol(k)
            bwd_choices[:amp => sk] = old_choices[:amp => sk]
        end
        # bwd_choices[:amp => :mu] = old_choices[:amp => :mu]
        # bwd_choices[:amp => :scale] = old_choices[:amp => :scale]
        # bwd_choices[:amp => :sigma] = old_choices[:amp => :sigma]
        # bwd_choices[:amp => :epsilon] = old_choices[:amp => :epsilon]
        
    elseif old_source_type == "harmonic" && new_source_type == "tone"
        
        new_gp_latents[:amp] = Dict()
        
        #trace[:source => source_id => (:element, i) => :erb] stays the same
        #Need to sample 1D amps
        sp = source_params["gp"]["amp"]["1D"]
        for k in keys(sp)
            sk = Symbol(k)
            new_gp_latents[:amp][sk] = @trace(sp[k]["dist"](sp[k]["args"]...), :amp => sk)
        end
        # new_gp_latents[:amp][:mu] = @trace(normal(50, 15), :amp => :mu)
        # new_gp_latents[:amp][:scale] = @trace(gamma(0.5,1), :amp => :scale) 
        # new_gp_latents[:amp][:sigma] = @trace(gamma(7,1), :amp => :sigma)
        # new_gp_latents[:amp][:epsilon] = @trace(gamma(1,1), :amp => :epsilon) 
        
        for i = 1:n_elements
            tfs, ts, fs = get_gp_spectrotemporal(old_abs_timings[i], steps, audio_sr)
            mu, cov = get_mu_cov(ts, new_gp_latents[:amp]) #TS for 1D tone
            new_choices[:source => source_id => (:element, i) => :amp] = @trace(mvnormal(mu, cov), (:element, i) => :amp)
            bwd_choices[(:element, i) => :amp] = old_choices[(:element, i) => :amp]
        end
        
        sp = source_params["gp"]["amp"]["2D"]
        for k in keys(sp)
            sk = Symbol(k)
            bwd_choices[:amp => sk] = old_choices[:amp => sk]
            # bwd_choices[:amp => :mu] = old_choices[:amp => :mu]
            # bwd_choices[:amp => :scale_t] = old_choices[:amp => :scale_t]
            # bwd_choices[:amp => :scale_f] = old_choices[:amp => :scale_f]
            # bwd_choices[:amp => :sigma] = old_choices[:amp => :sigma]
            # bwd_choices[:amp => :epsilon] = old_choices[:amp => :epsilon]
        end
        
    elseif old_source_type == "harmonic" && new_source_type == "noise"
        
        #FWD: Don't need to add anything

        sp = source_params["gp"]["erb"]
        for k in keys(sp)
            sk = Symbol(k)
            bwd_choices[:erb => sk] = old_choices[:erb => sk]
            # bwd_choices[:erb => :mu] = old_choices[:erb => :mu]
            # bwd_choices[:erb => :scale] = old_choices[:erb => :scale]
            # bwd_choices[:erb => :sigma] = old_choices[:erb => :sigma]
            # bwd_choices[:erb => :epsilon] = old_choices[:erb => :epsilon]
        end
        
        for i = 1:n_elements
            bwd_choices[(:element, i) => :erb] = old_choices[(:element, i) => :erb]
        end
        
    elseif old_source_type == "noise" && new_source_type == "harmonic"
        
        new_gp_latents[:erb] = Dict()

        #Need to add ERBs
        #trace[:source => source_id => (:element, i) => :amp] stays the same
        sp = source_params["gp"]["erb"]
        for k in keys(sp)
            sk = Symbol(k)
            new_gp_latents[:erb][sk] = @trace(sp[k]["dist"](sp[k]["args"]...), :erb => sk)
        end
        # new_gp_latents[:erb][:mu] = @trace(uniform(freq_to_ERB(20.0), freq_to_ERB(audio_sr/2. - 1.)), :erb => :mu)
        # new_gp_latents[:erb][:scale] = @trace(gamma(0.5,1), :erb => :scale) 
        # new_gp_latents[:erb][:sigma] = @trace(gamma(3,1), :erb => :sigma) 
        # new_gp_latents[:erb][:epsilon] = @trace(gamma(1,1), :erb => :epsilon) 
        
        for i = 1:n_elements
            tfs, ts, fs = get_gp_spectrotemporal(old_abs_timings[i], steps, audio_sr)
            mu, cov = get_mu_cov(ts, new_gp_latents[:erb])
            new_choices[:source => source_id => (:element, i) => :erb] = @trace(mvnormal(mu, cov), (:element, i) => :erb)
        end
        
    end
    
    for feature in keys(new_gp_latents)
        for attribute in keys(new_gp_latents[feature])
            new_choices[:source => source_id => feature => attribute] = new_gp_latents[feature][attribute]
        end
    end
    
    return new_choices, bwd_choices
        
end

function type_involution(trace, fwd_choices, fwd_ret, proposal_args)
    
    source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params = get_args(trace)
    new_choices, bwd_choices = fwd_ret
    new_trace, weight = update(trace, get_args(trace), (), new_choices)
    return new_trace, bwd_choices, weight
    
end