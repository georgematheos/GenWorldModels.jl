using Gen;
using GaussianProcesses;
using Dierckx;
include("../inference/proposals.jl")
include("../model/rendering.jl")
include("../model/model.jl")
include("../model/time_helpers.jl")
include("../model/gaussian_helpers.jl")
include("../tools/plotting.jl")
include("../inference/routine.jl")

function initialize_tone_sequences(elements, pad, steps)

    n_elements = length(elements)
    constraints = choicemap()
    constraints[:n_sources] = 1 
    constraints[:source => 1 => :source_type] = 1 #tone
    constraints[:source => 1 => :n_elements] = n_elements

    last_offset = pad[1]
    for i = 1:n_elements

        element = elements[i]
        
        wait = element["g"] + 1e-4*rand() + ( i == 1 ? last_offset : 0 )
        constraints[:source => 1 => (:element, i) => :wait ] = wait 
        duration = element["d"] + 1e-4*rand() 
        constraints[:source => 1 => (:element, i) => :dur_minus_min] = duration - steps["min"]
           
        onset = i == 1 ? wait : wait + last_offset
        offset = onset + duration 
        
        curr_t = get_element_gp_times([onset, offset], steps["t"])
        
        f = element["f"]
        erbSpl = Spline1D([onset,offset], [f,f], k=1)
        erbf0 = erbSpl(curr_t); 
        l = element["l"]
        ampSpl = Spline1D([onset,offset], [l,l], k=1)
        filt = ampSpl(curr_t);

        constraints[:source => 1 => (:element, i) => :erb] = erbf0
        constraints[:source => 1 => (:element, i) => :amp] = filt
        
        last_offset = offset 
        
    end
    
    scene_duration = last_offset + pad[2]
    return constraints, scene_duration
    
end

function tougas_bregman_1A()
    
    frequencies = [1600, 1270, 1008, 800, 635, 504, 400]
    levels = Dict("1600"=>75, "1270"=>73, "1008"=>70, "800"=> 70, "635"=> 70, "504"=> 72, "400"=>77)
    duration = 0.099
    gap = 0.001

    elements = []
    odd=1; even=1;
    for i = 1:13
        
        if i % 2 == 0
            f = frequencies[end:-1:1][odd]
            odd += 1
        else
            f = frequencies[even]
            even += 1
        end      
        element = Dict("f"=>freq_to_ERB(f), "l"=>levels[string(f)], 
                        "d"=>duration, "g"=>gap)
        push!(elements, element)
        
    end
    
    return elements
    
end

function bregman_rudnicky(standard, comparison, captor)
    
    #(0.1s sil)AB(1s sil)CCCDABDCC
    A = Dict("f" => 2200, "l" => 60, "t"=>"A")
    B = Dict("f" => 2400, "l" => 60, "t"=>"B")
    D = Dict("f" => 1460, "l" => 65, "t"=>"D")
    duration = 0.057
    gap_target = 0.008
    
    C = Dict( "far" => Dict("f" => 590, "l" => 63, "t"=>"C"),
              "mid" => Dict("f" => 1030, "l" => 60, "t"=>"C"),
              "near" => Dict("f" => 1460, "l" => 65, "t"=>"C"))
    gap_captors = 0.130 
    
    standard_tones = standard == "up" ? [A, B] : [B, A]
    captor_tones_before = captor == "none" ? [] : [C[captor], C[captor], C[captor]]
    comparison_tones = comparison == "up" ? [D, A, B, D] : [D, B, A, D]
    captor_tones_after = captor == "none" ? [] : [C[captor], C[captor]]
    
    elements = []
    for i = 1:length(standard_tones)
        tone = standard_tones[i]
        element = Dict("f"=>freq_to_ERB(tone["f"]), "l"=>tone["l"],
                    "d"=>duration,"g"=>gap_target);
        push!(elements, element)
    end
    
    separating_silence = 1.0
    if captor == "none"
        separating_silence += 3*(duration + gap_captors)
    else
        for i = 1:length(captor_tones_before)
            tone = captor_tones_before[i]
            gap = i == 1 ? separating_silence : gap_captors
            element = Dict("f"=>freq_to_ERB(tone["f"]), "l"=>tone["l"],
                        "d"=>duration,"g"=>gap);
            push!(elements, element)        
        end
    end
    
    for i = 1:length(comparison_tones)
        tone = comparison_tones[i]
        gap = i == 1 ? (captor == "none" ? separating_silence : gap_captors) : gap_target
        element = Dict("f"=>freq_to_ERB(tone["f"]), "l"=>tone["l"],
            "d"=>duration,"g"=>gap);
        push!(elements, element) 
    end
        
    if captor != "none"
        for i = 1:length(captor_tones_after)
            tone = captor_tones_after[i]
            element = Dict("f"=>freq_to_ERB(tone["f"]), "l"=>tone["l"],
                        "d"=>duration,"g"=>gap_captors);
            push!(elements, element)        
        end
    end
    
    return elements 
    
end

function ABA(semitones, onset_difference)
    
    #onset_difference: 60 - 800 ms
    #semitones: -15 to +15 
    A_freq = 1000
    B_freq = A_freq*(2. ^(semitones/12.));
    duration = 0.040
    g = onset_difference - duration
    level = 70
    reps = 5
    
    elements = []
    for r = 1:reps
        A_gap = g + ( r == 1 ? 0 : onset_difference )
        push!(elements, Dict("f"=>freq_to_ERB(A_freq), "l"=>level, 
                "g"=>A_gap, "d"=>duration))
        push!(elements, Dict("f"=>freq_to_ERB(B_freq), "l"=>level,
                            "g"=>g, "d"=>duration))
        push!(elements, Dict("f"=>freq_to_ERB(A_freq), "l"=>level,
                    "g"=>g, "d"=>duration))
    end
    
    return elements
    
end

function perfect_initialization(demofunc, demoargs;MLE=false,plot_trace=false,param_file="../params/base.jl",proposals=Dict())
    
    source_params, steps, gtg_params, obs_noise = include(param_file)
    constraints, scene_duration = initialize_tone_sequences(demofunc(demoargs...), [0.051, 0.050], steps)
#     println(constraints)
    audio_sr = 20000; 
    wts, f = gtg_weights(audio_sr, gtg_params)
    args = (source_params, Float64(scene_duration), wts, steps, Int(audio_sr), obs_noise, gtg_params)
    single_source_trace, _ = generate(generate_scene, args, constraints);
    scene_gram, t, scene_wave, source_waves, element_waves=get_retval(single_source_trace)
    plot_sources(single_source_trace,scene_gram,0;save=false)

    trace = single_source_trace
    for i = 1:50
        (fwd_choices, fwd_score, fwd_ret) = propose(switch_randomness, (trace,))
        (new_trace, bwd_choices, weight) = switch_involution(trace, fwd_choices, fwd_ret, ());
        if ~isinf(get_score(new_trace)) #randomize to a valid trace
            trace = new_trace
        end
    end
    scene_gram, t, scene_wave, source_waves, element_waves=get_retval(trace)
    plot_sources(trace,scene_gram,1;save=false)
     
    if MLE 
        for i = 1:trace[:n_sources]
 
            element_attributes = Dict(:wait => [], :dur_minus_min => [], :erb => [], :amp => [])
            for k in keys(element_attributes)
                compilefunc = k == :wait || k == :dur_minus_min ? push! : append! 
                for j = 1:trace[:source => i => :n_elements]
                    compilefunc(element_attributes[k], trace[:source => i => (:element, j) => k])
                end
            end
            abs_timings = absolute_timing(get_submap(get_choices(trace), :source => i), steps["min"])
            t = []
            for j = 1:trace[:source => i => :n_elements]
                append!(t, get_element_gp_times(abs_timings[j], steps["t"]))
            end

            for latent = [:erb, :amp]
                (proposal_trace, _) = Gen.generate(proposals[latent == :amp ? :amp1D : latent], (t,element_attributes[latent],));
                constraints = choicemap()
                constraints[:source => i => latent => :epsilon] = exp(proposal_trace[:gpparams][1])
                constraints[:source => i => latent => :mu] = proposal_trace[:gpparams][2]
                constraints[:source => i => latent => :scale] = exp(proposal_trace[:gpparams][3])
                constraints[:source => i => latent => :sigma] = exp(proposal_trace[:gpparams][4])
                (trace, w, _, discard) = update(trace, args, (), constraints)
            end
            for latent = [:wait, :dur_minus_min]
                (proposal_trace, _) = Gen.generate(proposals[latent], (element_attributes,));
                constraints = choicemap( (:source => i => latent => :mu, proposal_trace[:mu]),
                                         (:source => i => latent => :precision, proposal_trace[:precision]) )
                (trace, w, _, discard) = update(trace, args, (), constraints)
            end
            
        end
    end
    plot_sources(trace, scene_gram, 2; save=false)
    if plot_trace
        scene_gram, t, scene_wave, source_waves, element_waves=get_retval(trace)
        #plot_gtg(scene_gram, scene_duration, audio_sr/2, 20, 100)
        plot_sources(trace, scene_gram, 0; save=false)
    end
    
    return trace
    
end