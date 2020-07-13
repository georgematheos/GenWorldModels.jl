const lo_lim_freq = 20.0

#Frequency scale conversions
function freq_to_ERB(freq)
    return 9.265*log.(1 .+ freq./(24.7*9.265))
end

function ERB_to_freq(ERB)
    return 24.7*9.265*(exp.(ERB./9.265) .- 1)
end

function freq_to_octave(freq)
    return log2.(freq)
end

function octave_to_freq(ve)
    return 2 .^ve
end

#function get_tone_sample_times(tone_timing, tstep)
function get_element_gp_times(tone_timing, tstep)
    return [round(t, digits=3) for t in 0:tstep:tone_timing[2] if t >= tone_timing[1]]
end
    

#function get_tone_sample_freqs(audio_sr, fstep)
function get_element_gp_freqs(audio_sr, steps)                
    
    scale = steps["scale"]
    fstep = steps["f"]
                
    if scale == "ERB"
        freq_to_scale = freq_to_ERB
    elseif scale == "octave"
        freq_to_scale = freq_to_octave
    end
                
    lo = freq_to_scale(lo_lim_freq)
    hi = freq_to_scale(floor(audio_sr/2.) - 1.)
    
    return round.(range(lo, stop=hi, step=fstep), digits=3)
                
end
   
function get_gp_spectrotemporal(tone_timing, steps, audio_sr)                
    ts = get_element_gp_times(tone_timing, steps["t"])
    fs = get_element_gp_freqs(audio_sr, steps)    
    tfs = cross_tf(ts,fs)
    return tfs, ts, fs
end            
 
function cross_tf(ts, fs)
    tfs = []
    for t in ts
        for f in fs
            push!(tfs,[t,f])
        end
    end
    return tfs
end
            
                
function absolute_timing(choices, mindur; dream=false)
    #= Get a list of onset/offset pairs for a source =#

    n_elements = choices[:n_elements];
    total_time = 0; t = []
    for i = 1:n_elements
        if dream
            #check if the element has everything it needs to be full
            w = has_value(choices, (:element, i) => :wait)
            d = has_value(choices, (:element, i) => :dur_minus_min)
            a = has_value(choices, (:element, i) => :amp) #should be same as erb
            if ~w || ~d || ~a 
                #otherwise, these were elements that went beyond the scene duration 
                break
            end
        end
        onset = total_time + choices[(:element, i) => :wait]
        duration = choices[(:element, i) => :dur_minus_min] + mindur
        offset = onset + duration
        push!(t, [onset, offset])
        total_time += choices[(:element, i) => :wait] + choices[(:element, i) => :dur_minus_min] + mindur
    end
    return t
    
end