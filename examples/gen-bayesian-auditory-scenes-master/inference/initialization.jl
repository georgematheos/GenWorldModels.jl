using JSON;
using Dierckx;
using Random;
using WAV;
using Gen;


function load_sound(demo_name, gtg_params)
    sound_name = string("./sounds/", demo_name, ".wav")
    demo, audio_sr = wavread(sound_name);
    demo = demo[:,1];
    wts, gtg_freqs = gtg_weights(audio_sr, gtg_params)
    demo_gram, t = gammatonegram(demo, wts, audio_sr, gtg_params)
    scene_duration = length(demo)/audio_sr; 
    #plot_gtg(demo_gram, scene_duration, audio_sr/2)
    #title("$demo_name demo");
    return demo_gram, wts, scene_duration, audio_sr
end

function load_demo_sound(demo_name, gtg_params; dream_path="/om2/user/mcusi/gen-bayesian-auditory-scenes/dream/")

    dream_path = string(dream_path, "demos/")

    println("Loading demo generating parameters...")
    #Load parameters from demo generation
    demo_params = Dict()
    open(string(dream_path, "parameters.json"),"r") do f
        dt = read(f, String)
        demo_params = JSON.parse(dt)
    end
    gtg_params["ref"] = demo_params["rms_ref"] 

    demo, audio_sr = wavread(string(dream_path, demo_name, ".wav"));
    demo = demo[:,1];
    wts, gtg_freqs = gtg_weights(audio_sr, gtg_params) #ref doesn't affect the weights so this is fine!
    demo_gram, t = gammatonegram(demo, wts, audio_sr, gtg_params)
    scene_duration = length(demo)/audio_sr; 

    return demo_gram, wts, scene_duration, audio_sr

end

function read_guide(demo_name;demo_folder="./sounds/")
    #read in neural network guide mixture distribution
    d = Dict()
    f = open(string(demo_folder,demo_name, "_guide.json"), "r")
    dt = read(f,String)  # file information to string
    d = JSON.parse(dt)  # parse and transform data
    return d
end

function c2p(counts)
    return counts/sum(counts)
end
    
@gen function data_based_elements(d,mindur,source_params)
    elements = [];
    for i = 1:length(d["elements"])
        element = d["elements"][i]
        p_use = element["p_use"]["p"]
        use_element = p_use > 0.80 ? true : @trace(bernoulli(p_use), (:use, i))
        if use_element
            ps_to_use = element["source"]["ps"]
            ps_to_use = ps_to_use./sum(ps_to_use)   
            vs_to_use =  element["source"]["vs"]
            @assert vs_to_use == source_params["types"]
            source_type_idx = @trace(categorical(ps_to_use), (:source_type,i))#"tone"
            source_type = source_params["types"][source_type_idx]

            onset_ps = c2p(element["onset"][source_type]["ps"]); 
            onset_idx = @trace(categorical(onset_ps), (:onset, i))
            offset_ps = c2p(element["offset"][source_type]["ps"][onset_idx]); 
            offset_idx = @trace(categorical(offset_ps), (:offset, i))
            f0_idxs = []; filt_idxs = [];
            for j = onset_idx:offset_idx ## ITERATE OVER TIME INDEX 
                if source_type == "tone" || source_type == "harmonic"
                    f0_ps = c2p(element["f0"]["ps"][j]); 
                    next_f0_idx = @trace(categorical(f0_ps), (:f0, i, j))
                    push!(f0_idxs, next_f0_idx)
                end
                if source_type == "tone"
                    filt_ps = c2p(element["filt1D"]["ps"][j]);
                    next_filt_idx = @trace(categorical(filt_ps), (:filt, i, j))
                    push!(filt_idxs, next_filt_idx)
                end
                if source_type == "noise" || source_type == "harmonic"
                    timepoint_spectrum = element["filt2D"]["ps"][j]
                    for k = 1:length(timepoint_spectrum) ##ITERATE OVER FREQUENCY INDEX
                        filt_ps = c2p(timepoint_spectrum[k]);
                        next_filt_idx = @trace(categorical(filt_ps), (:filt, i, j, k)) #argmax(filt_ps)
                        push!(filt_idxs, next_filt_idx)
                    end
                end
            end
            
            onset = d["t_vs"][onset_idx] 
            offset = d["t_vs"][offset_idx]
            ts = d["t_vs"][onset_idx:offset_idx]
            filt = source_type == "tone" ? d["a_1Dvs"][filt_idxs] : d["a_2Dvs"][filt_idxs]; 
            erbf0 = source_type == "tone" || source_type == "harmonic" ? freq_to_ERB(d["f_vs"][f0_idxs]) : false 
            if offset - onset >= mindur
                push!(elements, Dict("source_type" => source_type, "source_type_idx"=>source_type_idx, "onset" => onset, "offset" => offset, "erbf0" => erbf0, "filt" => filt, "ts" => ts, "fs" => d["f_vs"]) )
            end
        end
    end
    
    
    for i = 1:length(elements)
        
        if i == 1 
            elements[i]["source"] = 1
        else
            ps = []
            for j = 1:source_params["n_sources"]["val"] #max_sources
                source_elems = [element for element in elements if haskey(element,"source")]
                source_elems = [element for element in source_elems if element["source"] == j]
                if length(source_elems) == 0
                    push!(ps, 0.1)
                elseif source_elems[1]["source_type"] == elements[i]["source_type"]
                                            
                    source_onsets = [se["onset"] for se in source_elems]
                    sorted_onset_idx = sortperm(source_onsets)
                    source_elems = source_elems[sorted_onset_idx]
                    this_onset = elements[i]["onset"]
                    this_offset = elements[i]["offset"]
                                
                    fits = false;
                    for k = 1:length(source_elems) + 1
                        if k == 1
                            fits = 0 < this_onset < this_offset < source_elems[k]["onset"]
                        elseif 1 < k <= length(source_elems) 
                            fits =  source_elems[k-1]["offset"] < this_onset < this_offset < source_elems[k]["onset"]
                        else 
                            fits = source_elems[k - 1]["offset"] < this_onset
                        end
                        if fits
                            break
                        end
                    end
                    push!(ps, fits ? 1 : 0)
               else
                    push!(ps, 0)
               end
            end
            #When the number of clusters in the neural net DPGMM increases,
            #you will need to increase the number of possible sources, so that it can fit everything it has
            #ArgumentError: Categorical: the condition isprobvec(p) is not satisfied.
            source_idx = @trace(categorical(ps/sum(ps)),(:n, i))
            elements[i]["source"] = source_idx
        end
        
    end
                        
    return elements 
end;

function make_data_constraints(guide_dict, demo_gram, steps, random_seed, source_params)

    Random.seed!(random_seed)
    trace = simulate(data_based_elements, (guide_dict,steps["min"],source_params,))
    elements = get_retval(trace);

    constraints = choicemap()
    sources = []
    for j = 1:source_params["n_sources"]["val"]
        source_elems = [element for element in elements if element["source"] == j]
        if length(source_elems) > 0
            source_onsets = [se["onset"] for se in source_elems]
            sorted_onset_idx = sortperm(source_onsets)
            source_elems = source_elems[sorted_onset_idx]
        end
        push!(sources, source_elems)
    end

    sources = [source for source in sources if length(source) > 0]
    constraints[:n_sources] = length(sources)
    for (i, source) in enumerate(sources)
        n_elements = length(source)
        constraints[:source => i => :n_elements] = n_elements
        constraints[:source => i => :source_type] = source[1]["source_type_idx"]

        last_offset = 0;
        for (j, element) in enumerate(source)
            #if you don't add random component, the neural network grid exactly aligns with 
            #the GP grid and floating point errors means there are errors in curr_t
            onset = element["onset"] + 1e-4*rand(); offset = element["offset"] + 1e-4*rand();

            constraints[:source => i =>  (:element, j) => :wait] = onset - last_offset
            constraints[:source => i => (:element, j) => :dur_minus_min] = offset - onset - steps["min"];
            if element["source_type"] == "tone" || element["source_type"] == "harmonic" 
                curr_t = get_element_gp_times([onset, offset], steps["t"])

                erbSpl = Spline1D(element["ts"], element["erbf0"], k=1)
                erbf0 = erbSpl(curr_t); 
                                                                            
                constraints[:source => i => (:element, j) => :erb] = erbf0
            end
            
            if element["source_type"] == "tone"
                curr_t = get_element_gp_times([onset, offset], steps["t"])
                ampSpl = Spline1D(element["ts"], element["filt"], k=1)
                filt = ampSpl(curr_t);
                constraints[:source => i => (:element, j) => :amp] = filt            
            end
                                                                                                    
            if element["source_type"] == "noise" || element["source_type"] == "harmonic"
                                                                            
                curr_tfs, curr_ts, curr_fs = get_gp_spectrotemporal([onset, offset], steps, audio_sr) 
                filt = reshape(element["filt"], (length(element["fs"]), length(element["ts"]))) #potentially need to tranpose??
                noiseSpl = Spline2D(freq_to_ERB(element["fs"]), element["ts"], filt; kx=1, ky=1)
                tile_f = repeat(curr_fs, outer=length(curr_ts))
                tile_t = repeat(curr_ts, inner=length(curr_fs))
                amp = noiseSpl(tile_f, tile_t) 
                constraints[:source => i => (:element, j) => :amp] = reshape(amp, (length(curr_fs)*length(curr_ts),))                                                            
                                                                            
            end
            last_offset = offset

       end
    end
    constraints[:scene] = demo_gram;
                                                                
    return constraints
                                                                
end
                                                            
function init_from_likelihood(n_samples, guide_dict, demo_gram, args)
    steps=args[4]
    source_params=args[1]
    obs_variance = 1.0
    log_likelihood = 0; data_init_trace = Dict(); 
    for i = 1:n_samples
        constraints = make_data_constraints(guide_dict, demo_gram, steps, i, source_params);
        new_data_init_trace, _ = generate(generate_scene, args, constraints);
        new_log_likelihood = get_score(new_data_init_trace)#logpdf(noisy_matrix,get_retval(new_data_init_trace)[1],demo_gram, obs_variance)
        if i == 1
            data_init_trace = new_data_init_trace
            log_likelihood = new_log_likelihood
        elseif new_log_likelihood > log_likelihood
            data_init_trace = new_data_init_trace
            log_likelihood = new_log_likelihood
        end
    end
                                                                                      
    return data_init_trace
end


function sample_inits(n_samples, guide_dict, demo_gram, args)
    steps=args[4]
    source_params=args[1]
    obs_variance = 1.0
    init_traces = []
    for i = 1:n_samples
        constraints = make_data_constraints(guide_dict, demo_gram, steps, i, source_params);
        new_data_init_trace, _ = generate(generate_scene, args, constraints);
        push!(init_traces, new_data_init_trace)
    end
                                                                                      
    return init_traces
end