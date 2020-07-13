using Gen
using GaussianProcesses
using Random
using Distributions
using JSON

using Statistics: mean, std, cor;
using LinearAlgebra: dot;
using StatsFuns: logsumexp, softplus;
using PyPlot
using SpecialFunctions: digamma,trigamma;
include("../model/time_helpers.jl")
include("../model/extra_distributions.jl")
include("../model/gaussian_helpers.jl")

function make_source_latent_model(source_params, audio_sr, steps)
    
    @gen function source_latent_model(latents, scene_duration)

        ##Single function for generating data for amoritized inference to propose source-level latents
        # Wait: can generate on its own
        # Dur_minus_min: can generate on its own
        # GPs:
        # Can have a separate amoritized inference move for ERB, Amp-1D, Amp-2D 
        # Need to sample wait and dur_minus_min to define the time points for the GP 
        #
        # Format of latents: 
        # latents = Dict(:gp => :amp OR :tp => :wait, :source_type => "tone")

        ### SOURCE-LEVEL LATENTS 
        ## Sample GP source-level latents if needed 
        gp_latents = Dict()
        if :gp in keys(latents)

            gp_params = source_params["gp"]
            gp_type = latents[:gp]; gp_latents[gp_type] = Dict();
            source_type = latents[:source_type]
            
            hyperpriors = gp_type == :erb ? gp_params["erb"] : 
                ((source_type == "noise" || source_type == "harmonic") ? gp_params["amp"]["2D"] : gp_params["amp"]["1D"] )
    
            for latent in keys(hyperpriors)
                hyperprior = hyperpriors[latent]; syml = Symbol(latent)
                gp_latents[gp_type][syml] = @trace(hyperprior["dist"](hyperprior["args"]...), gp_type => syml)
            end
            
        end

        ## Sample temporal source-level latents 
        tp_latents = Dict()
        if :tp in keys(latents)
            if typeof(latents[:tp]) == Symbol
                tp_latents[latents[:tp]] = Dict()
            else
                for a in latents[:tp]
                    tp_latents[a] = Dict()
                end
            end
        elseif :gp in keys(latents)
            tp_latents[:wait] = Dict()
            tp_latents[:dur_minus_min] = Dict()
        end            
        tp_params = source_params["tp"]
        for tp_type in keys(tp_latents)
            hyperpriors = tp_params[String(tp_type)]
            for latent in keys(hyperpriors)
                hyperprior = hyperpriors[latent]; syml = Symbol(latent)
                tp_latents[tp_type][syml] = @trace(hyperprior["dist"](hyperprior["args"]...), tp_type => syml)
            end
            tp_latents[tp_type][:args] = (tp_latents[tp_type][:a], tp_latents[tp_type][:mu]/tp_latents[tp_type][:a]) #a, b for gamma
        end

        ## Sample a number of elements 
        ne_params = source_params["n_elements"]
        n_elements = ne_params["type"] == "max" ? 
            @trace(uniform_discrete(1, ne_params["val"]),:n_elements) : 
            @trace(geometric(ne_params["val"]),:n_elements)    

        ### ELEMENT-LEVEL LATENTS 
        #Storage for what inputs are needed
        tp_elems = Dict( [ k => [] for k in keys(tp_latents)]... )
        gp_elems = Dict(); x_elems = [];
        if :gp in keys(latents)       
            
            gp_type = latents[:gp]; source_type = latents[:source_type]
            
            gp_elems[gp_type] = []
            gp_type = latents[:gp]
            gp_elems[:t] = []
            if gp_type == :amp && (source_type == "noise" || source_type == "harmonic")
                gp_elems[:reshaped] = []
                gp_elems[:f] = []
                gp_elems[:tf] = []
            end 
            
        end
                
        time_so_far = 0.0;
        for element_idx = 1:n_elements
            
            if :wait in keys(tp_elems)
                wait = element_idx == 1 ? @trace(uniform(0, scene_duration-steps["t"]), (:element,element_idx)=>:wait) : 
                    @trace(gamma(tp_latents[:wait][:args]...), (:element,element_idx)=>:wait)
                push!(tp_elems[:wait], wait)
            end
            
            if :dur_minus_min in keys(tp_elems)
                dur_minus_min = @trace(truncated_gamma(tp_latents[:dur_minus_min][:args]..., source_params["duration_limit"]), (:element,element_idx)=>:dur_minus_min); 
                push!(tp_elems[:dur_minus_min], dur_minus_min)
            end
            
            if :gp in keys(latents)
                
                gp_type = latents[:gp]; source_type = latents[:source_type]
                duration = dur_minus_min + steps["min"]; onset = time_so_far + wait; 

                if onset > scene_duration
                    break
                end

                time_so_far = onset + duration; element_timing = [onset, time_so_far]

                ## Define points at which the GPs should be sampled
                x = []; ts = [];
                if gp_type === :erb || (source_type == "tone" && gp_type === :amp)
                    x = get_element_gp_times(element_timing, steps["t"])
                elseif gp_type === :amp && (source_type == "noise"  || source_type == "harmonic")
                    x, ts, gp_elems[:f] = get_gp_spectrotemporal(element_timing, steps, audio_sr)
                end

                mu, cov = element_idx == 1 ? get_mu_cov(x, gp_latents[gp_type]) : 
                        get_cond_mu_cov(x, x_elems, gp_elems[gp_type], gp_latents[gp_type])
                element_gp = @trace(mvnormal(mu, cov), (:element, element_idx) => gp_type)
                
                ## Save the element data 
                append!(x_elems, x)
                if gp_type === :erb || (source_type == "tone" && gp_type === :amp)
                    append!(gp_elems[:t], x)
                    append!(gp_elems[gp_type], element_gp)
                elseif gp_type === :amp && (source_type == "noise"  || source_type == "harmonic")
                    append!(gp_elems[:t],ts)
                    append!(gp_elems[:tf],x) 
                    append!(gp_elems[gp_type], element_gp)
                    reshaped_elem = reshape(element_gp, (length(gp_elems[:f]), length(ts))) 
                    if element_idx == 1
                        gp_elems[:reshaped] = reshaped_elem
                    else
                        gp_elems[:reshaped] = cat(gp_elems[:reshaped], reshaped_elem, dims=2)
                    end
                end

                if time_so_far > scene_duration
                    break
                end
           
            end

        end

        return tp_latents, gp_latents, tp_elems, gp_elems

    end
 
    return source_latent_model
    
end

function make_data_generator(source_latent_model, latents)
    
    function data_generator()


        max_scene_duration = 2.5; min_scene_duration = 0.5;
        scene_duration = round((max_scene_duration-min_scene_duration)*rand() + min_scene_duration,digits=3)
        trace = simulate(source_latent_model, (latents,scene_duration))
        tp_latents, gp_latents, tp_elems, gp_elems = get_retval(trace)

        constraints = choicemap()
        if :tp in keys(latents)
            tp_latent = latents[:tp]
            constraints[tp_latent => :mu] = trace[tp_latent => :mu]
            constraints[tp_latent => :a] = trace[tp_latent => :a]
            
        elseif :gp in keys(latents)
            
            gp_type = latents[:gp]
            source_type = latents[:source_type]
            d = gp_type == :erb ? source_params["gp"]["erb"] : (source_type == "tone" ? source_params["gp"]["amp"]["1D"] : source_params["gp"]["amp"]["2D"]) 
            for k in keys(d)
                constraints[gp_type => Symbol(k)] = trace[gp_type => Symbol(k)]
            end
            
        end
        inputs = :tp in keys(latents) ? (tp_elems,) : (gp_elems,scene_duration,)
        
        return (inputs, constraints)

    end
    
    return data_generator
    
end;

function create_hmc_proposal(hyperpriors; n_prior_samples=1000, n_hmc_samples=5000) #n_chains  

    prior_args = Dict(); 
    for sv in keys(hyperpriors)
        if sv == "epsilon" || sv == "scale" || sv == "sigma"
            x = [ log(hyperpriors[sv]["dist"](hyperpriors[sv]["args"]...)) for g in 1:n_prior_samples ]
        elseif sv == "mu"
            x = [ hyperpriors[sv]["dist"](hyperpriors[sv]["args"]...) for g in 1:n_prior_samples ]
        end
        prior_args[sv] = [ mean(x), std(x) ];
        println("$sv: ", prior_args[sv])
    end

    @gen function hmc_based_proposal(x, y)
        
        #for i = 1:n_chains
        mConstant = GaussianProcesses.MeanConst(prior_args["mu"][1])
        kern = GaussianProcesses.SE(0.0, 0.0)
        logObsNoise = -1.0
        gp_ess = GaussianProcesses.GP(Float64.(x),Float64.(y),mConstant,kern, logObsNoise)

        GaussianProcesses.set_priors!(gp_ess.mean, [Distributions.Normal(prior_args["mu"]...)]) 
        GaussianProcesses.set_priors!(gp_ess.kernel, [Distributions.Normal(prior_args["scale"]...), Distributions.Normal(prior_args["sigma"]...)]) 
        GaussianProcesses.set_priors!(gp_ess.logNoise, [Distributions.Normal(prior_args["epsilon"]...)])

        rng = MersenneTwister(2143)
        chain = GaussianProcesses.ess(rng, gp_ess, nIter=n_hmc_samples)
        #end

        #["Log epsilon", "Mean", "SE log scale", "SE log sigma"]
        gp_params_mean = vec(mean(chain,dims=2))
        gp_params_cov = cov(chain,dims=2)
        GaussianProcesses.make_posdef!(gp_params_cov)[1]

        gp_params = @trace(mvnormal(gp_params_mean, gp_params_cov), :gpparams)

    end

    return hmc_based_proposal
end