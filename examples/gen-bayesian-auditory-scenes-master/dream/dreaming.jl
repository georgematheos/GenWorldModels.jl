using Gen
using Printf
using NPZ #to save numpy arrays
using JSON
using WAV
include("../model/gammatonegram.jl")
include("../model/time_helpers.jl")
include("../model/rendering.jl")
include("../model/model.jl")

function latent_outputs(trace, elements_to_keep, t_gtg, f_gtg, amp_classes_1D, amp_classes_2D, steps, gtg_params, audio_sr, source_params)
    
    #Dimensionality of outputs
    n_valid_elems = sum([sum(elems) for elems in elements_to_keep])
    n_source_types = 2; n_on_off_vars = 2;
    n_amp_classes_1D = length(amp_classes_1D)
    n_amp_classes_2D = length(amp_classes_2D)
    erb_gtg = freq_to_ERB(f_gtg); 
    
    #define output arrays
    #use false instead of Int64 zero in ordr to save disk space!  
    temporal_and_source = falses(length(t_gtg), n_source_types*n_on_off_vars, n_valid_elems)
    f0 = falses(length(t_gtg), length(f_gtg), n_valid_elems)
    filt1D = falses(length(t_gtg), n_amp_classes_1D, n_valid_elems)
    filt2D = falses(length(t_gtg), length(f_gtg), n_amp_classes_2D, n_valid_elems)
    mask = falses(length(t_gtg), n_valid_elems)
    
    #encode trace into 1/0 output arrays
    n_sources = trace[:n_sources]
    valid_idx = 1
    for source_idx = 1:n_sources
        source_trace = get_submap(get_choices(trace), :source => source_idx)
        #Neural network has to have everything in the same ordering as source_trace["types"]
        type_idx = source_trace[:source_type]  
        #n_elements = source_trace[:n_elements]
        tone_timing = absolute_timing(source_trace, steps["min"], dream=true)
        
        for elem_idx = 1:length(tone_timing)
            #get "n_elements" from here because there is an "early end" clause in 
            #the model generation script, so that the script doesn't have to waste time
            #generating elements that are off the page
            
            if elements_to_keep[source_idx][elem_idx] == 0 
                continue
            end
                
            ### 1. Source and temporal variables
            # Retrieve source type and purely temporal variables (onset, offset/duration)
            onset_idx = argmin([ abs(tone_timing[elem_idx][1] - tp) for tp in t_gtg])
            offset_idx = argmin([ abs(tone_timing[elem_idx][2] - tp) for tp in t_gtg])
            @assert onset_idx <= offset_idx
            temporal_and_source[onset_idx, type_idx, valid_idx] = true
            temporal_and_source[offset_idx, n_source_types + type_idx, valid_idx] = true
            #Retrieve various associated time vectors 
            tfe_latent, te_latent, fe_latent = get_gp_spectrotemporal(tone_timing[elem_idx], steps, audio_sr)
            onset_to_offset_idxs = onset_idx:offset_idx; 
            te_gtg = t_gtg[onset_to_offset_idxs]
            
            ### 2. Fundamental frequency
            if source_params["types"][type_idx] == "tone" || source_params["types"][type_idx] == "harmonic"
                
                #Rather than having the network estimate exactly the matrix sampled in webppl,
                #we are going to linearly interpolate that matrix to the size of the gammatonegram 
                #Reasoning: otherwise, it is finicky and confusing to make the network operations
                #           give the correct sized outputs; particularly due to batching
                #This is also reasonable because we linearly interpolate (in audio samples domain) to actually create a soundwave when rendering
                erb_latent = source_trace[(:element, elem_idx) => :erb]
                erbSpl = length(erb_latent) == 1 ? 
                    (f(x)=fill(erb_latent[1], size(x))) : Spline1D(te_latent, erb_latent, k=1)
                erbe_gtg = erbSpl(te_gtg); 
                
                ##Format for the network
                #Classify the sampled erb into the closest erb bin
                #Put a 1 in that bin and zeros for all the other frequency bins
                for tidx = 1:length(onset_to_offset_idxs)
                    fidx = argmin([ abs(erbe_gtg[tidx] - eg) for eg in erb_gtg])
                    a=onset_to_offset_idxs[tidx]
                    f0[onset_to_offset_idxs[tidx],fidx,valid_idx] = true
                end
            end 
            
            if source_params["types"][type_idx] == "noise" || source_params["types"][type_idx] == "harmonic"
                amp_latent = source_trace[(:element, elem_idx) => :amp]
                amp_latent = reshape(amp_latent, (length(fe_latent), length(te_latent)))
                if length(te_latent) == 1
                    specSpl = Spline1D(fe_latent, amp_latent[:, 1], k=1)
                    spec = specSpl(erb_gtg)
                    ampe_gtg = repeat(spec, 1, length(te_gtg))
                else
                    ampSpl = Spline2D(fe_latent, te_latent, amp_latent; kx=1, ky=1)
                    ampe_gtg = evalgrid(ampSpl, erb_gtg, te_gtg)
                end

                for fidx = 1:length(erb_gtg)
                    for tidx = 1:length(onset_to_offset_idxs)
                        ampe_val = ampe_gtg[fidx, tidx]
                        aidx = argmin([ abs(ampe_val - ac) for ac in amp_classes_2D]) 
                        filt2D[onset_to_offset_idxs[tidx],fidx,aidx,valid_idx] = true
                    end
                end
               
            elseif source_params["types"][type_idx] == "tone"
                #I don't know why it breaks to use amp_latent here instead of source_trace[(:element, elem_idx) => :amp]
                ampSpl = (length(source_trace[(:element, elem_idx) => :amp]) == 1 ? 
                    (f(x)=fill(source_trace[(:element, elem_idx) => :amp][1], size(x))) : Spline1D(te_latent, source_trace[(:element, elem_idx) => :amp], k=1))
                ampe_gtg = ampSpl(te_gtg); 
                for tidx = 1:length(onset_to_offset_idxs)
                    aidx = argmin([ abs(ampe_gtg[tidx] - ac) for ac in amp_classes_1D])         
                    filt1D[onset_to_offset_idxs[tidx],aidx,valid_idx] = true
                end

            end
            #Mask tells you which bins you should evaluate the net cost with
            for oo_idx in onset_to_offset_idxs
                mask[oo_idx, valid_idx] = true
            end    
            valid_idx += 1
        end   
    end

    latents = Dict("timing"=>temporal_and_source, "f0"=>f0, "filt1D"=>filt1D, 
                "filt2D"=>filt2D, "mask"=>mask)
        
    return latents
    
end


function compute_ideal_mask(element_grams, gtg_params)

#     input
#     -----
#     Sxxs: frequency by time by elements
#
#     output
#     -------
#     ideal binary masks (nElements, t, f)
    
    
    nElements = size(element_grams)[1]    
    #Find where all elements have lowerBound in the Gammatonegram
    element_thresh = element_grams .== gtg_params["dB_threshold"]
    silence=trues(1,size(element_grams)[2],size(element_grams)[3])
    all!(silence,element_thresh)
    
    #find the largest softmax output for each (t,f) bin and assign 1 to that element, and 0 for the rest
    #the argmax will return the first index if maximum value is repeated
    #in practice probably don't really have to worry about this because 
    #very unlikely that two floats will be the same
    #IBM is ideally a float32 
    IBM = falses(size(element_grams))
    vals, idxs = findmax(element_grams, dims=1)
    for idx in idxs
        IBM[idx] = true
    end
    #and in the case that that "Same value" is just the lowerbound, 
    #it'll show up in the first index, so we set those to zero
    IBM[1,:,:] = IBM[1,:,:] .* .!silence[1,:,:]

    return IBM
end
    

function dreaming(dataset_name, params_name, parallel, n_samples; dream_path="/om2/user/mcusi/gen-bayesian-auditory-scenes/dreams/", start_idx=1)

    parallel_name = @sprintf("%02d",parse(Int,parallel)); 
    dream_path = string(dream_path, dataset_name, "_", parallel_name, "/")
    gen_params = Dict()
    if isdir(dream_path)
        #it's possible that the slurm schedule cancels a run and then starts it up again
        #if path is already made, set start_idx appropriately so it doesn't overwrite samples 
        
        startidxfn = string(dream_path, "startidx.json")
        println("Continuing old run: ", startidxfn)
        open(startidxfn,"r") do f
            dt = read(f, String)
            start_idx = parse(Int, dt)
        end

        println("Loading generative parameters")
        params_file = string(dream_path, "params_",params_name,".jl")
        source_params, steps, gtg_params, obs_noise = include(params_file)
        
        open(string(dream_path, "genparams.json"),"r") do f
            dt = read(f, String)
            gen_params = JSON.parse(dt)
        end

        audio_sr = gen_params["audio_sr"]; 
        min_scene_duration = gen_params["dur_intv"][1]; 
        max_scene_duration = gen_params["dur_intv"][2]; 
        wts, f = gtg_weights(audio_sr, gtg_params)
        amp_classes_1D = gen_params["amps_1D"]
        amp_classes_2D = gen_params["amps_2D"]
        source_params["dream"] = true
        source_params["duration_limit"] = max_scene_duration

    else
        
        println("Making ", dream_path)
        mkdir(dream_path)

        println("Defining source parameters...")
        params_file = string("../params/",params_name,".jl")
        source_params, steps, gtg_params, obs_noise = include(params_file)        
        println("Saving generative parameters...")
        cp(params_file, string(dream_path, "params_",params_name,".jl"))
        
        println("Defining scene parameters...")
        audio_sr = 20000; 
        min_scene_duration = 0.5;
        max_scene_duration = 2.0;
        wts, f = gtg_weights(audio_sr,gtg_params);
        source_params["dream"] = true
        source_params["duration_limit"] = max_scene_duration

        #Based on investigating the marginal distributions of the GPs beforehand.
        amp_classes_1D = gtg_params["dB_threshold"]:3:gtg_params["dB_threshold"]+80.0
        amp_classes_2D = range(-15.0, length=length(amp_classes_1D), stop=65.0)

        println("Saving scene parameters...")
        gen_params = Dict("gtg_params"=>gtg_params, "audio_sr"=>Int(audio_sr), "dur_intv"=>[min_scene_duration,max_scene_duration], "amps_1D"=>amp_classes_1D, "amps_2D"=>amp_classes_2D, "f"=>f, "n_amps"=>length(amp_classes_1D),"source_types"=>source_params["types"])
        open(string(dream_path, "genparams.json"),"w") do f
            JSON.print(f, gen_params)
        end
    
    end

    println("Start idx: ", start_idx)
    #check if there is an old sets that you didn't get finished
    println("Checking for leftovers...")
    dream_files = readdir(dream_path)
    startidx_name = @sprintf("%08d",start_idx); 
    strchk(s) = occursin(startidx_name, s)
    kfs = filter(strchk, dream_files)
    if length(kfs) >= 8
        println("Too many of: ", startidx_name) 
        for kf in kfs
            rm(string(dream_path, kf))
        end
    elseif 0 < length(kfs) < 8
        println("Too few of: ", startidx_name)
        for kf in kfs
            rm(string(dream_path, kf))
        end
    end

    println("Starting sampling process!")
    for sample_idx = start_idx:start_idx+n_samples
        
        scene_duration = round((max_scene_duration-min_scene_duration)*rand() + min_scene_duration,digits=3)
        args = (source_params, float(scene_duration), wts, steps, Int(audio_sr), obs_noise, gtg_params)

        #Generate a sample
        trace, = generate(generate_scene, args);
        scene_gram, t, scene_wave, source_waves, element_waves = get_retval(trace)
        #Make sure this is a float32:
        scene_gram = permutedims(scene_gram, (2, 1)) #scene_gram: t,f
        t_len = size(scene_gram)[1];
        @assert t_len == length(t)
        f_len = size(scene_gram)[2];
        @assert f_len == length(f)
               
        ###Get all cochleagrams and masks
        element_grams = Array{Float32}(undef, 0, t_len, f_len) #nElements t f 
        elements_to_keep = []
        for i = 1:length(element_waves)
            source_elements_to_keep = []
            for j = 1:size(element_waves[i])[2]
                gtg, t = gammatonegram(element_waves[i][:,j],wts,audio_sr,gtg_params)
                gtg = reshape(gtg, (f_len, t_len, 1))
                gtg = permutedims(gtg, [3, 2, 1])  #nElements, t, f
                element_grams = i == 1 && j == 1 ? gtg : cat(element_grams, gtg; dims=1)
                push!(source_elements_to_keep, all(gtg .== gtg_params["dB_threshold"]) ? 0 : 1)
            end
            push!(elements_to_keep, source_elements_to_keep)
        end
        tokeep(x) = x == 1
        idx_to_keep = findall(tokeep, vcat(elements_to_keep...))
        if length(idx_to_keep) == 0
            open(string(dream_path, "startidx.json"),"w") do f
                next_sample_idx = sample_idx + 1
                JSON.print(f, next_sample_idx)
            end
            continue
        end
        element_grams = element_grams[idx_to_keep, :, :]
        IBM = compute_ideal_mask(element_grams, gtg_params)
        
        #normalize scene and element_grams 
        # scene_gram = (scene_gram .- gtg_params["dB_threshold"])./gtg_params["normalization"]
        # element_grams = (element_grams .- gtg_params["dB_threshold"])./gtg_params["normalization"] 
        element_grams = permutedims(element_grams, [2, 3, 1]) #(nElements, t, f) --> (t, f, nElements)
        ideal_masks = permutedims(IBM, [2, 3, 1]) #(nElements, t, f) --> (t, f, nElements)
        
        ###Organize latent variables as neural net outputs 
        output_data = latent_outputs(trace, elements_to_keep, t, f, amp_classes_1D, amp_classes_2D, steps, gtg_params, audio_sr, source_params)
        input_data = Dict("scene" => Float32.(scene_gram),"elems" => Float32.(element_grams), "ims"=>ideal_masks);
        #All of output data is bools
        #ims are bools
        #this will save us space on /om2/
        #create_tf_record should load them in as the correct type needed for tensorflow!

        ###Save all the outputs
        sample_name = @sprintf("%08d",sample_idx); 
        nel_name = @sprintf("%03d",length(idx_to_keep));
        fn_start = string(dream_path,sample_name,"_");
        fn_end = string("_",nel_name,".npy");
        for data in [output_data, input_data]
            for k in keys(data)
                npzwrite(string(fn_start,k,fn_end), data[k])
            end
        end
        #"timing","f0","filt1D","filt2D","mask"
        #"scene","elems","ibm"

        open(string(dream_path, "startidx.json"),"w") do f
            next_sample_idx = sample_idx + 1
            JSON.print(f, next_sample_idx)
        end
        
    end
    println("Complete.")
end


function hearing(dataset_name; dream_path="/om2/user/mcusi/gen-bayesian-auditory-scenes/dreams/")

    dream_path = string(dream_path, "demos/")

    println("Loading demo generating parameters...")
    #Load parameters from demo generation
    demo_params = Dict()
    open(string(dream_path, "parameters.json"),"r") do f
        dt = read(f, String)
        demo_params = JSON.parse(dt)
    end

    println("Loading dataset generating parameters...")
    tfrecord_path = "/om/user/mcusi/dcbasa/data/"
    dataset_path = string(tfrecord_path, dataset_name,"_params.json")
    #Load parameters from Gen model (what the dataset was trained with)
    gen_params = Dict()
    open(dataset_path,"r") do f
        dt = read(f, String)
        gen_params = JSON.parse(dt)
    end

    if gen_params["audio_sr"] != demo_params["sr"]
        error("incompatible sampling rates")
    end

    gtg_params = gen_params["gtg_params"]
    gtg_params["ref"] = demo_params["rms_ref"] 

    demo_files = readdir(dream_path)
    strchk(s) = occursin(".wav", s)
    demo_files = filter(strchk, demo_files)

    wts, gtg_freqs = gtg_weights(gen_params["audio_sr"], gtg_params)
    println("Start generating gammatonegrams!")
    for demo_file in demo_files

        println(demo_file)
        #Load sound into Gen
        demo, audio_sr = wavread(string(dream_path, demo_file));
        demo = demo[:,1];
        demo_gram, t = gammatonegram(demo, wts, gen_params["audio_sr"], gtg_params)
        demo_gram = permutedims(Float32.(demo_gram), (2, 1)) #scene_gram: t,f
        npy_name = string(dream_path, split(demo_file, ".")[1], "_scene.npy")
        npzwrite(npy_name, demo_gram)
        
    end
    println("Complete.")
end