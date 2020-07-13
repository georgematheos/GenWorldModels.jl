using Gen
using GaussianProcesses
using GaussianMixtures
using Random
using Distributions
using JSON

using Statistics: mean, std, cor;
using LinearAlgebra: dot, transpose;
using StatsFuns: logsumexp, softplus;
using PyPlot
using SpecialFunctions: digamma,trigamma;
include("../model/time_helpers.jl")
include("../model/extra_distributions.jl")
include("../model/gaussian_helpers.jl")
include("../tools/debug_helpers.jl")
include("../tools/perfect_init.jl")

using Logging
debug_gm_print = false
logging_idx = debug_gm_print ? -1 : 0
Logging.disable_logging(LogLevel(logging_idx))

println("Defining gaussian process proposals")
function define_gp_proposal(hyperpriors; n_prior_samples=1000, n_hmc_samples=5000) #n_chains  

    prior_args = Dict(); 
    for sv in keys(hyperpriors)
        if sv == "epsilon" || sv == "scale" || sv == "sigma"
            x = [ log(hyperpriors[sv]["dist"](hyperpriors[sv]["args"]...)) for g in 1:n_prior_samples ]
        elseif sv == "mu"
            x = [ hyperpriors[sv]["dist"](hyperpriors[sv]["args"]...) for g in 1:n_prior_samples ]
        end
        prior_args[sv] = [ mean(x), std(x) ];
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
        chain = GaussianProcesses.ess(rng, gp_ess, nIter=n_hmc_samples, burn=1000)
        #=["Log epsilon", "Mean", "SE log scale", "SE log sigma"]
        gp_params_mean = vec(mean(chain,dims=2))
        gp_params_cov = cov(chain,dims=2)
        GaussianProcesses.make_posdef!(gp_params_cov)[1]
        gp_params = @trace(mvnormal(gp_params_mean, gp_params_cov), :gpparams)
        =#
        
#         idx = @trace(uniform_discrete(1,size(chain)[2]), :idx)
#         c = zeros(4,4)
#         c[1,1]=0.001
#         c[2,2]=0.001
#         c[3,3]=0.001
#         c[4,4]=0.00001
#         gpparams = @trace(mvnormal(chain[:,idx], c), :gpparams)
        
        mm_data = transpose(chain)
        n_clusters = 5
        gmm_chain = GaussianMixtures.GMM(n_clusters, mm_data; method=:kmeansdet, 
            kind=:full, nInit=20, nIter=15, nFinal=10, rng_seed=1, loglevel=logging_idx)
        # type GMM
        #     n::Int                         # number of Gaussians
        #     d::Int                         # dimension of Gaussian
        #     w::Vector                      # weights: n
        #     μ::Array                       # means: n x d
        #     Σ::Union(Array, Vector{Array}) # diagonal covariances n x d, or Vector n of d x d full covariances
        #     hist::Array{History}           # history of this GMM
        # end
        cs = zeros(n_clusters, 4, 4)
        for cluster_idx=1:n_clusters
            cs[cluster_idx,:,:] = GaussianMixtures.covar(gmm_chain.Σ[cluster_idx])
        end
        gp_params = @trace(mvn_mixture(gmm_chain.μ,cs,gmm_chain.w),:gpparams)

    end

    return hmc_based_proposal
end

println("Defining temporal proposals")
function define_tp_proposal(latent, source_params)
    
    @gen function conjugate_prior_proposal(tp_elems)

        #data
        w = tp_elems[latent]
        n = length(w)

        #hyperpriors
        mu_0 = source_params["mu"]["args"][1]
        kappa_0 = source_params["mu"]["args"][2]
        alpha_0 = source_params["precision"]["args"][1]
        beta_0 = source_params["precision"]["args"][2]
        
        if latent == :wait && n > 2 || latent == :dur_minus_min
        
            #w ~ log normal distribution 
            x = log.(latent == :wait ? w[2:end] : w)
            mu_x = mean(x)

            # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf Eq 85 - 89
            mu_n = ( kappa_0*mu_0 + n*mu_x )/(kappa_0 + n)
            kappa_n = kappa_0 + n
            alpha_n = alpha_0 + 0.5*n 
            # take 1/murphy because Murphy is using rate and we need to use scale for Gen's Gamma distribution 
            beta_n = 1 / ( 1/beta_0 + 0.5*sum( (x .- mu_x).^2 ) + 0.5*(kappa_0*n*(mu_x - mu_0)^2)/(kappa_0 + n)  )

            precision = @trace(gamma(alpha_n, beta_n), :precision)
            sigma = 1/sqrt(precision)
            mu = @trace(normal(mu_n, sigma/kappa_n), :mu)
        
        else
            
            precision = @trace(gamma(alpha_0, beta_0), :precision)
            sigma = 1/sqrt(precision)
            mu = @trace(normal(mu_0, sigma/kappa_0), :mu)
            
        end

    end
    
    return conjugate_prior_proposal
    
end

println("compiling proposals")
source_params, steps, gtg_params, obs_noise = include("../params/gnprior.jl")
audio_sr = 20000;

proposals = Dict()
proposals[:wait] = define_tp_proposal(:wait, source_params["tp"]["wait"])
proposals[:dur_minus_min] = define_tp_proposal(:dur_minus_min, source_params["tp"]["dur_minus_min"])
proposals[:erb] = define_gp_proposal(source_params["gp"]["erb"])
proposals[:amp1D] = define_gp_proposal(source_params["gp"]["amp"]["1D"])

println("defining switch randomness and involution")
@gen function rewrite_switch_randomness(trace, custom_proposals)
    
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
    switch_to_new_source = (old_n_elements > 1 && old_n_sources < source_params["n_sources"]["val"]) ? 1 : 0 ## currently hard coded that we're using a uniform distribution
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
    
    ## change the source level variables to increase probabilities of acceptance
    source_type = source_params["types"][old_choices[:source => origin => :source_type]] #because we retain the source_type for switches
    tp_types = collect(keys(source_params["tp"]))
    gp_types = source_type == "tone" || source_type == "harmonic" ? ["erb", "amp"] : ["amp"]
    source_vars = append!(tp_types, gp_types)

    #tp: wait & dur_minus_min --> a (variability), mu (mean)
    #gp: erb --> mu, scale, sigma, noise
    #    amp --> mu, scale, sigma, noise OR mu, scale_t, scale_f, sigma, noise

    dest_elements_list = Dict(:t=>[])
    for source_var in source_vars 
        #Collecting the description of all the elements in the destination stream
        source_var_sym = Symbol(source_var)
        dest_elements_list[source_var_sym] = []
        compilefunc = source_var == "erb" || source_var == "amp" ? append! : push! #push for waits & dur_minus_min, they're scalars
        if ~new_source
            k = 1
            for j = 1:old_choices[:source => destination => :n_elements] + 1
                if j == which_spot[destination] ##Get the elements in order.
                    compilefunc(dest_elements_list[source_var_sym], old_choices[:source => origin => (:element, element_idx) => source_var_sym])
                else
                    compilefunc(dest_elements_list[source_var_sym], old_choices[:source => destination => (:element, k) => source_var_sym])
                    k += 1
                end
            end
        else
            compilefunc(dest_elements_list[source_var_sym], old_choices[:source => origin => (:element, element_idx) => source_var_sym])
        end
    end

    #Get a time vector 
    if ~new_source
        k = 1
        for j = 1:old_choices[:source => destination => :n_elements] + 1
            if j == which_spot[destination] ##Get the elements in order.
                append!(dest_elements_list[:t], get_element_gp_times(all_source_timings[origin][element_idx], steps["t"]))
            else
                append!(dest_elements_list[:t], get_element_gp_times(all_source_timings[destination][k], steps["t"]))
                k += 1
            end
        end
    else
        append!(dest_elements_list[:t], get_element_gp_times(all_source_timings[origin][element_idx], steps["t"]))
    end

    for source_var in source_vars 
        source_var_sym = Symbol(source_var)
        if source_var == "wait" || source_var == "dur_minus_min"
            @trace(custom_proposals[source_var_sym](dest_elements_list), :dest => source_var_sym)
        elseif source_var == "erb" || source_var == "amp"
            proposal_key = source_var == "erb" ? :erb : (source_type == "tone" ? :amp1D : :amp2D)
            @trace(custom_proposals[proposal_key](dest_elements_list[:t],dest_elements_list[source_var_sym]), :dest => source_var_sym)       
        end
    end
    #Only works for tones right now...                                                                                                  
    
    if old_choices[:source => origin => :n_elements] > 1  
        #there must be an still existing origin source
        origin_elements_list = Dict(:t=>[])
        for source_var in source_vars 
            #Collecting the description of all the elements in the destination stream
            source_var_sym = Symbol(source_var)
            origin_elements_list[source_var_sym] = []
            compilefunc = source_var == "erb" || source_var == "amp" ? append! : push! #push for waits & dur_minus_min, they're scalars
            for j = [jj for jj in 1:old_choices[:source => origin => :n_elements] if jj != element_idx]
                compilefunc(origin_elements_list[source_var_sym], old_choices[:source => origin => (:element, j) => source_var_sym])
            end
        end
        for j = [jj for jj in 1:old_choices[:source => origin => :n_elements] if jj != element_idx]
            append!(origin_elements_list[:t], get_element_gp_times(all_source_timings[origin][j], steps["t"]))
        end

        #Only works for tones!!
        for source_var in source_vars 
            source_var_sym = Symbol(source_var)
            if source_var == "wait" || source_var == "dur_minus_min"
                @trace(custom_proposals[source_var_sym](origin_elements_list), :orig => source_var_sym)
            elseif source_var == "erb" || source_var == "amp"
                proposal_key = source_var == "erb" ? :erb : (source_type == "tone" ? :amp1D : :amp2D)
                @trace(custom_proposals[proposal_key](origin_elements_list[:t],origin_elements_list[source_var_sym]), :orig => source_var_sym)       
            end
        end
                                    
    end
                                                                                                                                                                                                                                      
    return which_spot, all_source_timings
               
end

function rewrite_switch_involution(trace, fwd_choices, fwd_ret, proposal_args)

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
    gp_hyperpriors = source_type == "tone" ? Dict(:erb => source_params["gp"]["erb"], :amp => source_params["gp"]["amp"]["1D"]) :
                     (source_type == "harmonic" ? Dict(:erb => source_params["gp"]["erb"], :amp => source_params["gp"]["amp"]["2D"]) :
                      Dict(:amp => source_params["gp"]["amp"]["2D"]) ) #for noise
    gp_types = keys(gp_hyperpriors)
    tp_types = [:wait, :dur_minus_min]
    source_attributes_list = []
    for (k, v) in gp_hyperpriors
        push!(source_attributes_list, k => [Symbol(kv) for kv in keys(v)])
    end
    for k in tp_types 
        push!(source_attributes_list, k => [Symbol(kv) for kv in keys(source_params["tp"][String(k)])])
    end
    source_attributes = Dict(source_attributes_list...)
    element_attributes = append!(tp_types, collect(gp_types))
    element_attributes_no_wait = append!([:dur_minus_min], gp_types)

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
        for ks in keys(source_attributes)
            if ks == :erb || ks == :amp
                #Only works for 1D right now!!
                new_choices[:source => new_destination_idx => ks => :mu] = fwd_choices[:dest => ks => :gpparams][2]
                new_choices[:source => new_destination_idx => ks => :sigma] = exp(fwd_choices[:dest => ks => :gpparams][4])
                new_choices[:source => new_destination_idx => ks => :epsilon] = exp(fwd_choices[:dest => ks => :gpparams][1])
                new_choices[:source => new_destination_idx => ks => :scale] = exp(fwd_choices[:dest => ks => :gpparams][3])
            else #temporal attributes, weight and durminsmin
                for a in source_attributes[ks]
                    new_choices[:source => new_destination_idx => ks => a] = fwd_choices[:dest => ks => a]
                end
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
        for ks in keys(source_attributes)
            if ks == :erb || ks == :amp
                #Only works for 1D gps!
                g1 = log(old_choices[:source => origin_idx => ks => :epsilon])
                g2 = old_choices[:source => origin_idx => ks => :mu]
                g3 = log(old_choices[:source => origin_idx => ks => :scale])
                g4 = log(old_choices[:source => origin_idx => ks => :sigma])

                bwd_choices[:dest => ks => :gpparams] = vec([g1, g2, g3, g4])

                g1 = log(old_choices[:source => destination_idx => ks => :epsilon])
                g2 = old_choices[:source => destination_idx => ks => :mu]
                g3 = log(old_choices[:source => destination_idx => ks => :scale])
                g4 = log(old_choices[:source => destination_idx => ks => :sigma])

                bwd_choices[:orig => ks => :gpparams] = vec([g1, g2, g3, g4])
            else #temporal attributes, weight and durminsmin
                for a in source_attributes[ks]
                    bwd_choices[:dest => ks => a] = old_choices[:source => origin_idx => ks => a]
                    bwd_choices[:orig => ks => a] = old_choices[:source => destination_idx => ks => a]
                end
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
        for ks in keys(source_attributes)
            if ks == :erb || ks == :amp
                #Only works for 1D right now!!
                new_choices[:source => destination_idx => ks => :mu] = fwd_choices[:dest => ks => :gpparams][2]
                new_choices[:source => destination_idx => ks => :sigma] = exp(fwd_choices[:dest => ks => :gpparams][4])
                new_choices[:source => destination_idx => ks => :epsilon] = exp(fwd_choices[:dest => ks => :gpparams][1])
                new_choices[:source => destination_idx => ks => :scale] = exp(fwd_choices[:dest => ks => :gpparams][3])
            else #temporal attributes, weight and durminsmin
                for a in source_attributes[ks]
                    new_choices[:source => destination_idx => ks => a] = fwd_choices[:dest => ks => a]
                end
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
        for ks in keys(source_attributes)
            if ks == :erb || ks == :amp
                #Only works for 1D right now!!
                new_choices[:source => new_origin_idx => ks => :mu] = fwd_choices[:orig => ks => :gpparams][2]
                new_choices[:source => new_origin_idx => ks => :sigma] = exp(fwd_choices[:orig => ks => :gpparams][4])
                new_choices[:source => new_origin_idx => ks => :epsilon] = exp(fwd_choices[:orig => ks => :gpparams][1])
                new_choices[:source => new_origin_idx => ks => :scale] = exp(fwd_choices[:orig => ks => :gpparams][3])
            else #temporal attributes, weight and durminsmin
                for a in source_attributes[ks]
                    new_choices[:source => new_origin_idx => ks => a] = fwd_choices[:orig => ks => a]
                end
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
        for ks in keys(source_attributes)
            if ks == :erb || ks == :amp
                #Only works for 1D gps!
                g1 = log(old_choices[:source => origin_idx => ks => :epsilon])
                g2 = old_choices[:source => origin_idx => ks => :mu]
                g3 = log(old_choices[:source => origin_idx => ks => :scale])
                g4 = log(old_choices[:source => origin_idx => ks => :sigma])

                bwd_choices[:dest => ks => :gpparams] = vec([g1, g2, g3, g4])

            else #temporal attributes, weight and durminsmin
                for a in source_attributes[ks]
                    bwd_choices[:dest => ks => a] = old_choices[:source => origin_idx => ks => a]
                end
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
        for ks in keys(source_attributes)
            if ks == :erb || ks == :amp
                #Only works for 1D right now!!
                new_choices[:source => origin_idx => ks => :mu] = fwd_choices[:orig => ks => :gpparams][2]
                new_choices[:source => origin_idx => ks => :sigma] = exp(fwd_choices[:orig => ks => :gpparams][4])
                new_choices[:source => origin_idx => ks => :epsilon] = exp(fwd_choices[:orig => ks => :gpparams][1])
                new_choices[:source => origin_idx => ks => :scale] = exp(fwd_choices[:orig => ks => :gpparams][3])
            else #temporal attributes, weight and durminsmin
                for a in source_attributes[ks]
                    new_choices[:source => origin_idx => ks => a] = fwd_choices[:orig => ks => a]
                end
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
        for ks in keys(source_attributes)
            if ks == :erb || ks == :amp
                #Only works for 1D right now!!
                new_choices[:source => destination_idx => ks => :mu] = fwd_choices[:dest => ks => :gpparams][2]
                new_choices[:source => destination_idx => ks => :sigma] = exp(fwd_choices[:dest => ks => :gpparams][4])
                new_choices[:source => destination_idx => ks => :epsilon] = exp(fwd_choices[:dest => ks => :gpparams][1])
                new_choices[:source => destination_idx => ks => :scale] = exp(fwd_choices[:dest => ks => :gpparams][3])
            else #temporal attributes, weight and durminsmin
                for a in source_attributes[ks]
                    new_choices[:source => destination_idx => ks => a] = fwd_choices[:dest => ks => a]
                end
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
        for ks in keys(source_attributes)
            if ks == :erb || ks == :amp
                #Only works for 1D gps!
                g1 = log(old_choices[:source => origin_idx => ks => :epsilon])
                g2 = old_choices[:source => origin_idx => ks => :mu]
                g3 = log(old_choices[:source => origin_idx => ks => :scale])
                g4 = log(old_choices[:source => origin_idx => ks => :sigma])

                bwd_choices[:dest => ks => :gpparams] = vec([g1, g2, g3, g4])

                g1 = log(old_choices[:source => destination_idx => ks => :epsilon])
                g2 = old_choices[:source => destination_idx => ks => :mu]
                g3 = log(old_choices[:source => destination_idx => ks => :scale])
                g4 = log(old_choices[:source => destination_idx => ks => :sigma])

                bwd_choices[:orig => ks => :gpparams] = vec([g1, g2, g3, g4])
            else #temporal attributes, weight and durminsmin
                for a in source_attributes[ks]
                    bwd_choices[:dest => ks => a] = old_choices[:source => origin_idx => ks => a]
                    bwd_choices[:orig => ks => a] = old_choices[:source => destination_idx => ks => a]
                end
            end
        end
                                            
              
    end
    new_trace, weight = update(trace, get_args(trace), (), new_choices)
    return new_trace, bwd_choices, weight

end

println("generating demo trace")
demofunc = tougas_bregman_1A
demoargs = [] #empty
demoargs = Tuple(demoargs)
demo_trace = perfect_initialization(demofunc,demoargs;MLE=true,param_file="../params/gnprior.jl",proposals=proposals);
demo_gram, _, _, _, _ = get_retval(demo_trace);
plot_sources(demo_trace, demo_gram, 0; save=true, save_loc="../sounds/randinit_newprior/")

println("beginning inference")
trace = demo_trace
#traces = [trace]
for i = 1:15000
	global trace
    trace, accepted = mh(trace, rewrite_switch_randomness, (proposals,), rewrite_switch_involution) 
    if accepted
        plot_sources(trace, demo_gram, i; save=true, save_loc="../sounds/randinit_newprior/")
    end
    #push!(traces, trace)
end
