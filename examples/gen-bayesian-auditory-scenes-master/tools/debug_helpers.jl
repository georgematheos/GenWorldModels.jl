using Gen
using Random
using JSON
using Statistics: mean, std, cor;

using PyPlot
include("../model/time_helpers.jl")
include("../model/gaussian_helpers.jl")

#= Set of Gen functions 
mainly to help debug inference 
some other tools in here too (eg module loc)

test_involution_move: test that the proposal goes correctly forward and backward
check_switch_proposal_probs (for switch proposals only): check how likely each choice from that proposal was
run_proposal: run a stochastic involution proposal and see if it is likely to be accepted
test_proposal_choices: constrain some of the forward choices

see also plotting.jl
plot_sources(new_trace, demo_gram, i; save=false)

=# 

function test_involution_fixed_randomness_args(trace, randomness, randomness_args, involution)
	#= Compares the initial trace and the "round trip" trace
	They should have the same sampled variables (within floating point error)
	And the same scores (within floating point error) =#

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
        
        for svar = [:erb, :amp]
            for a = [:mu, :sigma, :scale, :epsilon]
                @assert(isapprox(source_trace[svar => a], source_trace_round_trip[svar => a]),
                    string("Value-- $svar $a: ", source_trace[svar => a]," not approx ",source_trace_round_trip[svar => a]))
                s1 = project(trace, select(:source => source_idx => svar => a));  
                s2 = project(trace_round_trip, select(:source => source_idx => svar => a));
                @assert(isapprox(s1, s2), "Score-- $svar $a $s1 not approx $s2")
            end
        end
        
        for svar = [:wait, :dur_minus_min]
            for a = [:mu, :a]
                @assert(isapprox(source_trace[svar => a], source_trace_round_trip[svar => a]),["$svar $a: ", source_trace[svar => a]," not approx ",source_trace_round_trip[svar => a]])
                s1 = project(trace, select(:source => source_idx => svar => a));  s2 = project(trace_round_trip, select(:source => source_idx => svar => a));
                @assert(isapprox(s1, s2), "Score-- $svar $a $s1 not approx $s2")
            end
        end
        
        element_as = [:wait, :dur_minus_min, :erb, :amp] 
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
	#= 
	args are the trace args 
	randomness_args_fn should return a list of tuples with args for the involution 
	=#

    # generate a random trace of the model
    trace = simulate(model, args) 
    # get args for ranadoness
    randomness_args_list = randomness_args_fn(trace)
    for randomness_args in randomness_args_list
        test_involution_fixed_randomness_args(trace, randomness, randomness_args, involution)
    end
    return "Successful test!"

end


function check_switch_proposal_probs(randomness, randomness_args, new_or_init_trace, bwd_or_fwd_choices)
    #=
    This function checks on the probability of each choice inside a switch proposal
    randomness: a gen function
    bwd_choices should pair with new_trace (bwd moves away from new_trace to init_trace)
    fwd_choices should pair with init_trace (fwd moves away from init_trace to new_trace)
    =#

    (choice_trace, _) = generate(randomness, (new_or_init_trace,randomness_args...), bwd_or_fwd_choices);
    println("Choice trace:")
    println(get_choices(choice_trace))
    println()
    #Switch 
    println("New source, score: ", project(choice_trace, select(:new_source)))
    println("Element idx, score: ", project(choice_trace, select(:element_idx)))
    println("Origin, score: ", project(choice_trace, select(:origin)))
    println("Destination, score: ", project(choice_trace, select(:destination)))
    #Hyperparameter choices
    for location in [:orig, :dest]
        println(location)
        for gp in [:erb, :amp]
            println("$gp score: ", project(choice_trace, select(location => gp => :gpparams)))
        end
        for tp in [:wait, :dur_minus_min]
            for a in [:precision, :mu]
                println("$tp $a score: ", project(choice_trace, select(location => tp => a)))
            end
        end
    end

end

function run_involution(randomness, involution, randomness_args, init_trace; print_choices=false)

    (fwd_choices, fwd_score, fwd_ret) = propose(randomness, (init_trace, randomness_args...))
    (new_trace, bwd_choices, weight) = involution(init_trace, fwd_choices, fwd_ret, ())
    (bwd_score, _) = assess(randomness, (new_trace,randomness_args...), bwd_choices)
    a = weight - fwd_score + bwd_score
    println("Weight: $(weight) - Fwd: $(fwd_score) + Bwd: $(bwd_score) = $a")
    if print_choices
        println("Forward choices:")
        println(fwd_choices)
        println("Backward choices")
        println(bwd_choices)
    end
    return fwd_choices, bwd_choices, new_trace

end

function check_score(trace;trace_name="")
    
    #= Use this function to print out the score 
    associated with each choice in the trace,
    comparing the inital and final traces 
    =#  
    total = 0
    for i = 1:trace[:n_sources]
        single = project(trace, select(:source => i))
        total += single
        println("$trace_name Trace -- source $i: $single")

        for source_var in [:wait, :dur_minus_min, :erb, :amp]
            if source_var == :wait || source_var == :dur_minus_min
                sp = source_params["tp"][String(source_var)]
            elseif source_var == :erb
                sp = source_params["gp"]["erb"]
            elseif source_var == :amp
                sp = source_params["gp"]["amp"]["1D"]
            end
            println("\t$source_var")
            for a in keys(sp)
                sa = Symbol(a)
                v = round(trace[:source => i => source_var => sa], digits=6)
                rp = round(project(trace, select(:source => i => source_var => sa)), digits=5)
                println("\t\t $a=$v: ", rp)
            end
        end

        n_tones = trace[:source => i => :n_elements]
        println("\tn_tones $n_tones: ", project(trace, select(:source => i => :n_elements)))
        for j = 1:trace[:source => i => :n_elements]
            println("\tTone $j")
            for a = [:wait, :dur_minus_min, :erb, :amp]
                mval = (a==:erb||a==:amp) ? mean(trace[:source => i => (:element, j) => a]) : trace[:source => i => (:element, j) => a]
                v = round(mval, digits=6)
                rp = round(project(trace, select(:source => i => (:element, j) => a)),digits=5)
                println("\t\t $a=$v: ", rp)
            end
        end
    end
    println("$trace_name Trace -- total: $total");println("")

end


function compare_score_by_var(initial_trace, new_trace, obs_gram; switch_proposal=false)
    
    #= Use this function to print out the score 
    associated with each choice in the trace,
    comparing the inital and final traces 
    =#

    println("Looking at each part of the score:");println("")
    noise_value = 1.0 #initial_trace[:noise]
    println("Using fixed noise value = $noise_value")
    # old_scene, _, _ = get_retval(initial_trace)
    # new_scene, _, _ = get_retval(new_trace)
    ## NOTE: if you use "old_scene" or "new_scene" instead of the value from the trace, the likelihood does NOT come out equal!!
    old_likelihood=Gen.logpdf(noisy_matrix,initial_trace[:scene],obs_gram,noise_value)
    new_likelihood=Gen.logpdf(noisy_matrix,new_trace[:scene],obs_gram,noise_value)
    println("Old likelihood: $old_likelihood, New likelihood: $new_likelihood")
    if switch_proposal
        @assert old_likelihood == new_likelihood
        println("Likelihoods are equal.")
    end

    println(""); println("Score of random choices:")
    check_score(initial_trace;trace_name="Initial")
    check_score(new_trace;trace_name="New")

end

function test_proposal_choices(randomness, involution, randomness_args, init_trace, fwd_contraints;print_choices=false)

    (fwd_trace, _) = generate(randomness, (init_trace,randomness_args...), fwd_contraints);
    fwd_choices = get_choices(fwd_trace); fwd_score = get_score(fwd_trace); fwd_ret = get_retval(fwd_trace)
    (new_trace, bwd_choices, weight) = involution(init_trace, fwd_choices, fwd_ret, ())
    (bwd_score, _) = assess(randomness, (new_trace,randomness_args...), bwd_choices)
    a = weight - fwd_score + bwd_score
    println("Weight: $(weight) - Fwd: $(fwd_score) + Bwd: $(bwd_score) = $a")
    if print_choices
        println("Forward choices:")
        println(fwd_choices)
        println("Backward choices")
        println(bwd_choices)
    end
    return fwd_choices, bwd_choices, new_trace
    
end

function plot_gaussian_process_funcs(trace, steps; ylimits=Dict(:erb=>[5,30],:amp=>[60,85]))
    
    ts = get_element_gp_times([0,get_args(trace)[2]], steps["t"])
    gp_params = [ Dict() for i = 1:trace[:n_sources] ]
    for i = 1:trace[:n_sources]
        for latent in [:erb, :amp]
            gp_params[i][latent] = Dict()
            for k in [:mu, :sigma, :scale, :epsilon]
                gp_params[i][latent][k] = trace[:source => i => latent => k]
            end
        end
    end

    plt.figure(figsize=(20,5*trace[:n_sources]))
    
    for i = 1:trace[:n_sources]
        for (lidx, latent) in enumerate([:erb,:amp])
            subplot(trace[:n_sources], 2, 2*(i-1) + lidx )
            abs_times = absolute_timing(get_submap(get_choices(trace),(:source=>i)), steps["min"])
            t_y = cat([get_element_gp_times(abs_times[j], steps["t"]) for j = 1:trace[:source=>i=>:n_elements]]...,dims=1)
            y = cat([trace[:source=>i=>(:element,j)=>latent] for j=1:trace[:source=>i=>:n_elements]]...,dims=1)
            t_x = ts
            mu, C = get_cond_mu_cov(t_x, t_y, y, gp_params[i][latent])

            for nf = 1:50
                f = Gen.random(mvnormal, mu, C)
                plot(t_x,f,alpha=0.2)
            end 
            xlim([0,get_args(trace)[2]])
            ylim(ylimits[latent])
            scatter(t_y,y)
            mu = round(gp_params[i][latent][:mu],digits=4)
            eps = round(gp_params[i][latent][:epsilon],digits=4)
            sig = round(gp_params[i][latent][:sigma],digits=4)
            scl = round(gp_params[i][latent][:scale],digits=4)
            title("Latent: $latent, src: $i, mu:$mu, sig:$sig, scale:$scl, eps:$eps, ")
        end
    end
    plt.tight_layout()
    
end

function moduleloc(M::Module)
    for n in names(M)
        if isdefined(M, n) && getfield(M, n) isa Function
            f = getfield(M, n)
            ms = Iterators.filter(m-> m.module==M, methods(f))
            if !isempty(ms)
                dir = join(split(string(first(ms).file), "/")[1:end-1], "/")
                return dir
            end
        end
    end
    return ""
end
