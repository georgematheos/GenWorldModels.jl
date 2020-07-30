using Gen;
include("../model/extra_distributions.jl")
include("../model/gammatonegram.jl")
include("../model/rendering.jl")
include("../model/time_helpers.jl")
include("../model/gaussian_helpers.jl")


@gen function sample_element_gps(source_type, element_idx, element_timing, gp_latents, prev_gps, scene_duration, steps, audio_sr, rms_ref, source_params)

    ## Define points at which the GPs should be sampled
    gps = Dict()
    if source_type == "tone"
        gps[:erb_x] = get_element_gp_times(element_timing, steps["t"])
        gps[:amp_x] = gps[:erb_x]
    elseif source_type == "harmonic" 
        gps[:amp_x], gps[:erb_x], fs = get_gp_spectrotemporal(element_timing, steps, audio_sr)
    elseif source_type == "noise" 
        gps[:amp_x], ts, fs = get_gp_spectrotemporal(element_timing, steps, audio_sr) 
    end

    ## Sample the GPs in a sequential manner
    features = source_type == "noise" ? [:amp] : [:erb, :amp]
    for feature in features
        feature_x = feature == :amp ? :amp_x : :erb_x
        mu, cov = element_idx == 1 ? get_mu_cov(gps[feature_x], gp_latents[feature]) : 
            get_cond_mu_cov(gps[feature_x], prev_gps[feature_x], prev_gps[feature], gp_latents[feature])
        gps[feature] = @trace(mvnormal(mu, cov), (:element, element_idx) => feature)
    end
    
    ## Generate the waveform given these element GPs
    element_duration = element_timing[2] - element_timing[1];
    if source_type == "tone"
        element_wave = generate_tone(gps[:erb], gps[:amp], element_duration, steps["t"], audio_sr, rms_ref)
    elseif source_type == "noise"
        amp = transpose(reshape(gps[:amp], (length(fs), length(ts))))
        element_wave = generate_noise(amp, element_duration, steps, audio_sr, rms_ref)
    elseif source_type == "harmonic"
        amp = transpose(reshape(gps[:amp], (length(fs), length(gps[:erb]))))
        element_wave = generate_harmonic(gps[:erb], amp, element_duration, source_params["n_harmonics"], steps, audio_sr, rms_ref)
    end
    
    ## Place the waveform into the overall scene register
    n_samples = Int(floor(scene_duration*audio_sr)); scene_wave = zeros(n_samples);
    onset = element_timing[1]; offset = minimum([element_timing[2], scene_duration]);
    sample_start = max(1, Int(floor(onset*audio_sr)));
    if sample_start < n_samples
        sample_finish = min(sample_start + length(element_wave), length(scene_wave))
        scene_wave[sample_start:sample_finish-1] = element_wave[1:length(sample_start:sample_finish-1)]
    end
    
    return scene_wave, gps
    
end

@gen function sample_source_latents(source_params)

    ###Sample source-level parameters
    ##Source type
    n_types = length(source_params["types"])
    source_type_idx = @trace(categorical(ones(n_types)/n_types),:source_type) 
    #@trace(bernoulli(0.5), :source_type) ? "tone" : "noise"
    source_type = source_params["types"][source_type_idx]

    ##Element number
    ne_params = source_params["n_elements"]
    n_elements = ne_params["type"] == "max" ? 
        @trace(uniform_discrete(1, ne_params["val"]), :n_elements) : 
        @trace(geometric(ne_params["val"]), :n_elements)
    
    ##Element spacing 
    tp_latents = Dict(:wait=>Dict(),:dur_minus_min=>Dict())
    tp_params = source_params["tp"]
    if "precision" in keys(tp_params["wait"])
        #Using Gamma-Normal prior for LogNormal wait and duration times 
        for tp_type in keys(tp_latents)
            hyperpriors = tp_params[String(tp_type)]
            hyperprior = hyperpriors["precision"]
            tp_latents[tp_type][:precision] = @trace(hyperprior["dist"](hyperprior["args"]...), tp_type => :precision)
            sigma = 1.0/sqrt(tp_latents[tp_type][:precision])
            hyperprior = hyperpriors["mu"]
            tp_latents[tp_type][:mu] = @trace(hyperprior["dist"](hyperprior["args"][1], sigma/sqrt(hyperprior["args"][2])), tp_type => :mu)
            tp_latents[tp_type][:dist] = tp_type == :wait ? log_normal : truncated_log_normal
            tp_latents[tp_type][:args] = (tp_latents[tp_type][:mu], sigma) 
        end
    else
        for tp_type in keys(tp_latents)
            hyperpriors = tp_params[String(tp_type)]
            for latent in keys(hyperpriors)
                hyperprior = hyperpriors[latent]; syml = Symbol(latent)
                tp_latents[tp_type][syml] = @trace(hyperprior["dist"](hyperprior["args"]...), tp_type => syml)
            end
            tp_latents[tp_type][:dist] = tp_type == :wait ? gamma : truncated_gamma
            tp_latents[tp_type][:args] = (tp_latents[tp_type][:a], tp_latents[tp_type][:mu]/tp_latents[tp_type][:a]) #a, b for gamma
        end
    end

    ##GPs 
    gp_latents = source_type == "noise" ? Dict(:amp => Dict()) : Dict(:erb => Dict(), :amp => Dict());
    gp_params = source_params["gp"]
    for gp_type in keys(gp_latents)
        hyperpriors = gp_type === :erb ? gp_params["erb"] : 
            ((source_type == "noise" || source_type == "harmonic") ? gp_params["amp"]["2D"] : gp_params["amp"]["1D"] )
        for latent in keys(hyperpriors)
            hyperprior = hyperpriors[latent]; syml = Symbol(latent)
            gp_latents[gp_type][syml] = @trace(hyperprior["dist"](hyperprior["args"]...), gp_type => syml)
        end
    end

    return source_type, n_elements, tp_latents, gp_latents

end

@gen function generate_source(source_params, scene_duration, steps, audio_sr, gtg_params)
    
    ###Get source-level latents
    source_type, n_elements, tp_latents, gp_latents = @trace(sample_source_latents(source_params))
    prev_gps = source_type == "noise" ? Dict(:amp => [], :amp_x => []) : 
        Dict(:amp => [], :amp_x => [], :erb => [], :erb_x => [])    

    ###Sample and render each tone
    n_samples = Int(floor(scene_duration*audio_sr))
    element_waves = zeros(n_samples, n_elements)    
    time_so_far = 0;
    #the other way to sample this would be sequentially and stop if wait > scene.
    for element_idx=1:n_elements
        
        wait = element_idx == 1 ? @trace(uniform(0.0, scene_duration), (:element,element_idx)=>:wait) : @trace(tp_latents[:wait][:dist](tp_latents[:wait][:args]...), (:element,element_idx)=>:wait)
        #For dur, use truncation otherwise a random sample can attempt to generate a sound that is very long and crash.
        dur_minus_min = @trace(tp_latents[:dur_minus_min][:dist](tp_latents[:dur_minus_min][:args]..., source_params["duration_limit"]), (:element,element_idx)=>:dur_minus_min); 
        duration = dur_minus_min + steps["min"]; onset = time_so_far + wait; time_so_far = onset + duration; element_timing = [onset, time_so_far]

        if onset > scene_duration && source_params["dream"]
            break
        end
        element_waves[:,element_idx], element_gps = @trace(sample_element_gps(source_type, element_idx, element_timing, gp_latents, prev_gps, scene_duration, steps, audio_sr, gtg_params["ref"], source_params)); 
        for k in keys(prev_gps)
            append!(prev_gps[k], element_gps[k]);
        end

        if time_so_far > scene_duration && source_params["dream"]
            break
        end

    end
    
    # sum over all tones to produce source waveform
    source_wave = sum(element_waves, dims=2)[:,1]
    return source_wave, element_waves 

end
generate_sources = Map(generate_source)


@gen function generate_scene(source_params, scene_duration, wts, steps, audio_sr, obs_noise, gtg_params)

    #Sample sources
    ns_params = source_params["n_sources"]
    n_sources = ns_params["type"] == "max" ? @trace(uniform_discrete(1, ns_params["val"]), :n_sources) : @trace(geometric(ns_params["val"]), :n_sources)
    waves = @trace(generate_sources(fill(source_params,n_sources),fill(scene_duration,n_sources),fill(steps,n_sources),fill(audio_sr,n_sources),fill(gtg_params,n_sources)),:source)
    source_waves, element_waves = collect(zip(waves...)) 
    n_samples = Int(floor(scene_duration*audio_sr))
    scene_wave = reduce(+, source_waves; init=zeros(n_samples))
    # generate spectrogram from waveform
    scene_gram, t = gammatonegram(scene_wave, wts, audio_sr, gtg_params) 
                    
    # add observation noise: either have noise as a random variable or anneal. 
    noise = obs_noise["type"] == "rand" ? @trace(exponential(obs_noise["val"]), :obs_noise) : obs_noise["val"]
    @trace(noisy_matrix(scene_gram, noise), :scene)
    
    return scene_gram, t, scene_wave, source_waves, element_waves
    
end