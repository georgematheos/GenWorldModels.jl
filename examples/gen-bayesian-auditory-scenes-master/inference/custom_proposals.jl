using PyCall
using GenTF
tf = pyimport("tensorflow")
tf.compat.v1.disable_eager_execution()

using Gen;
using Random;
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
            tp_latents[latents[:tp]] = Dict()
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
        inputs = :tp in keys(latents) ? (tp_elems,scene_duration,) : (gp_elems,scene_duration,)
        
        return (inputs, constraints)

    end
    
    return data_generator
    
end;


function create_trainable_tp_proposal(latent)
    
    @gen function trainable_tp_proposal(tp_elems, scene_duration)

        @param B_mean_mu::Vector{Float64}
        @param B_shape_mu::Float64
        @param C_shape_mu::Float64
        @param B_mean_alpha::Vector{Float64}
        @param B_shape_a::Float64
        @param C_shape_a::Float64
        @param MLE_estimate_alpha_0::Float64
        @param MOM_estimate_alpha_0::Float64
        
        n_elements = length(tp_elems[latent])
        if ( latent == :wait && n_elements > 1 ) || ( latent == :dur_minus_min )
            
            #For parameter estimation in Gamma distributions
            #MLE estimators             
            W = latent == :wait ? tp_elems[latent][2:end] : tp_elems[latent]
            MLE_estimate_mu = mean(W)
            if length(W) > 2    
                s = log(mean(W)) - mean(log.(W))
                MLE_estimate_alpha = (3.0 - s + sqrt((s-3)^2 + 24*s))/(12.0 * s) # initial guess
                for i = 1:5 #iterative updates based on Newton-Raphson method
                    numerator = log(MLE_estimate_alpha) - digamma(MLE_estimate_alpha) - s
                    denominator = (1.0/MLE_estimate_alpha) - trigamma(MLE_estimate_alpha)
                    MLE_estimate_alpha = MLE_estimate_alpha - (numerator/denominator)  
                end
            else
                MLE_estimate_alpha =  exp(MLE_estimate_alpha_0)
            end
            
            #Method of moments estimators 
            # MOM_estimate_mu = mean(W)
            # MOM_alpha is pretty close to MLE_Estimate_alpha
            MOM_estimate_alpha = length(W) > 2 ? (mean(W)^2)/(std(W)^2) : exp(MOM_estimate_alpha_0)

            X_mu = [log(MLE_estimate_mu), 1.0]
            X_alpha = [log(MLE_estimate_alpha), 
                     log(MOM_estimate_alpha),  
                     1.0]            
                        
            mean_mu_estimate = exp( dot(X_mu, B_mean_mu) )
            shape_mu = exp( B_shape_mu*log(n_elements) + C_shape_mu)
            mean_alpha_estimate = exp( dot(X_alpha, B_mean_alpha) )
            shape_a = exp( B_shape_a*log(n_elements) + C_shape_a )

            @trace(gamma(shape_mu, mean_mu_estimate/shape_mu), latent => :mu)
            @trace(gamma(shape_a, mean_alpha_estimate/shape_a), latent => :a)
            
        else
            
            @trace(log_normal(-2.0,1.5), latent => :mu)
            @trace(gamma(20,1), latent => :a)
            
        end

        return nothing

    end
    
    return trainable_tp_proposal
    
end

# function plot_hist_vs_tpprior(proposal, latent, vartoplot, tp_val, source_params; save_loc="./")

#     h=[]; n=length(tp_val)
#     for i = 1:1000
#         max_scene_duration = 2.5; min_scene_duration = 0.5;
#         scene_duration = round((max_scene_duration-min_scene_duration)*rand() + min_scene_duration,digits=3)
#         choices,_,=propose(proposal,(Dict(latent=>tp_val),scene_duration,))
#         push!(h,choices[latent => vartoplot])
#     end
#     hist(h,bins=20,density=true)
#     sp = source_params["tp"][string(latent)][string(vartoplot)]
#     xvals = vartoplot == :mu ? (0:20) : (0:50)
#     plot(xvals,exp.([Gen.logpdf(sp["dist"],x,sp["args"]...) for x in xvals]))
#     legend(["pdf","samples"])
#     title("Hist $vartoplot with $n tp point vs. prior pdf")
#     plt.savefig(string(save_loc, string(latent), "_", string(vartoplot) ,"__hist_vs_prior.png"))
#     plt.close()

# end

tf.compat.v1.reset_default_graph()
function define_neural_net(filter_height, filter_width, out_channels, n_params; batch_size=1, in_width=1, dilations=[1,2,4], final_regression="independent")
    
    #scene_t = get_element_gp_times([0,scene_duration], steps["t"])
    in_channels=2; 
    in_height=nothing;#none;#could use scene_duration as input to "define neural net" then this equals length(scene_t); 
    
    gp = tf.compat.v1.placeholder(dtype=tf.float64, 
        shape=(batch_size, in_height, in_width, in_channels)) #tf.placeholder
    mask = tf.compat.v1.placeholder(dtype=tf.float64, shape=(batch_size, in_height, 1, 1))

    #Defining network
    #filter_height = 3; filter_width = 1; out_channels = 16; n_params = 8;
    #dilations = [1,2,4]
    weights = []; layer_inputs = [gp];
    for i = 0:length(dilations)-1
        ic = i == 0 ? in_channels : out_channels
        oc = i == length(dilations) - 1 ? n_params : out_channels
        W = tf.compat.v1.get_variable("W$i", 
                (filter_height, filter_width, ic, oc), 
                dtype=tf.float64, 
                initializer=tf.compat.v1.initializers.glorot_normal())
        b = tf.compat.v1.get_variable("b$i", (oc), 
                dtype=tf.float64, 
                initializer=tf.constant_initializer(value=0.0))
        append!(weights, [W, b])
        L1 = tf.compat.v1.nn.convolution(layer_inputs[i+1], W, padding="SAME", 
             dilation_rate=(dilations[i+1], 1), data_format="NHWC")
        #L1 = tf.nn.conv2d(gp, W, (1,), padding="SAME", data_format="NHWC")
        L1 = tf.nn.bias_add(L1, b, data_format="NHWC")
        C = tf.nn.relu(L1)
        push!(layer_inputs, C)
    end
    #Convert maske convolutions to parameter estimates
    tile_mask = tf.tile(mask, (1, 1, 1, n_params))
    masked_conv = tf.multiply(tile_mask, layer_inputs[end])
    O = tf.reduce_sum(masked_conv,axis=(1,2)) #batch_size, n_params,
    denom = tf.expand_dims(tf.reduce_sum(mask, axis=(1,2,3)),1)
    O = tf.divide( O , denom ) 
    O = tf.nn.relu(O)
    if final_regression == "independent"
        Wf = tf.compat.v1.get_variable("Wfinal",(1,n_params),dtype=tf.float64,
            initializer=tf.compat.v1.initializers.glorot_normal())
        b = tf.compat.v1.get_variable("bfinal",(n_params),dtype=tf.float64, 
            initializer=tf.constant_initializer(value=0.0)) #put back to 1 and train longer?
        Wf_tile = tf.tile(Wf, (batch_size, 1))
        dist_parameters = tf.nn.bias_add(tf.multiply(Wf_tile,O),b)
    else
        Wf = tf.compat.v1.get_variable("Wfinal",(1, n_params, n_params),dtype=tf.float64,
            initializer=tf.compat.v1.initializers.glorot_normal()) 
        b = tf.compat.v1.get_variable("bfinal",(n_params),dtype=tf.float64, 
            initializer=tf.constant_initializer(value=0.0))
        Wf_tile = tf.tile(Wf, (batch_size, 1, 1))
        M = tf.matmul(Wf_tile, tf.expand_dims(O,2))
        dist_parameters = tf.nn.bias_add(tf.squeeze(M,2), b)
    end
    append!(weights, [Wf, b])

    for w in weights
        println(w)
    end

    nn = GenTF.TFFunction(weights, [gp, mask], dist_parameters);
    
    return nn, weights
    
end

function create_trainable_gp1D_proposal(latent, nn)

    @gen function trainable_gp1D_proposal(gp_elems, scene_duration)

        #format GP for input into neural network 
        scene_t = get_element_gp_times([0,scene_duration], steps["t"]) #If training on variable length scenes, change this
        mask = [t in gp_elems[:t] ? 1.0 : 0.0 for t in scene_t]
        embedded_gp = []; k = 1; 
        m = mean(gp_elems[latent]); 
        for t_idx in 1:length(scene_t)
            if mask[t_idx] == 1
                push!(embedded_gp, gp_elems[latent][k])
                k += 1
            else
                push!(embedded_gp, m)
            end
        end
        embedded_gp = embedded_gp .- m #should this be scaled? 
        net_input = cat(embedded_gp, mask, dims=2)
        mask = cat(mask...,dims=1)
        batch_size=1; in_channels=2; in_height=length(scene_t); in_width=1;
        net_input = float.(reshape(net_input, batch_size, in_height, in_width, in_channels))
        net_mask = float.(reshape(mask, batch_size, in_height, 1, 1))
        dparams = @trace(nn(net_input,net_mask), :net_parameters)
        
        mean_mu_estimate = dparams[1]*m; shape_mu = softplus(dparams[2])
        # println("mean_mu_estimate: $mean_mu_estimate")
        # println("shape_mu: $shape_mu")
        @trace(normal(mean_mu_estimate, shape_mu), latent => :mu) 
              
        mean_sigma_estimate = softplus(dparams[3]) + (length(gp_elems[latent]) > 1 ? std(gp_elems[latent]) : 0.0); 
        shape_sigma = softplus(dparams[4])
        # println("mean_sigma_estimate: $mean_sigma_estimate")
        # println("shape_sigma: $shape_sigma")
        @trace(gamma(shape_sigma, mean_sigma_estimate/shape_sigma), latent => :sigma)

        mean_scale_estimate = softplus(dparams[5]); shape_scale = softplus(dparams[6]); 
        # println("mean_scale_estimate: $mean_scale_estimate")
        # println("shape_scale: $shape_scale")
        @trace(gamma(shape_scale, mean_scale_estimate/shape_scale), latent => :scale)        

        mean_epsilon_estimate = softplus(dparams[7]); shape_epsilon = softplus(dparams[8]); 
        # println("mean_epsilon_estimate: $mean_epsilon_estimate")
        # println("shape_epsilon: $shape_epsilon")
        @trace(gamma(shape_epsilon, mean_epsilon_estimate/shape_epsilon), latent => :epsilon)

        return nothing

    end

    return trainable_gp1D_proposal
    
end

function print_neural_var(net_func, var_name)
    sess = get_session(net_func)
    V = [var for var in tf.compat.v1.global_variables() if var.op.name==var_name][1]
    println(var_name, ": ",net_func[:run](V))
end

function get_net_saver(net_func)
    saver = tf.compat.v1.train.Saver(Dict(string(var.name) => var for var in Gen.get_params(net_func)))
    return saver
end

function save_net_weights(saver, net_func, ckpt_name; save_loc="./")
    saver.save(GenTF.get_session(net_func), string(save_loc, ckpt_name, ".ckpt"))
end

function load_net_weights(saver, net_func, ckpt_name; save_loc="./")
    saver.restore(GenTF.get_session(net_func), string(save_loc, ckpt_name, ".ckpt"))
end

function plot_scores(latent,scores;save_loc="./")
    
    scatter(1:length(scores),scores,marker=".")
    xlabel("Iterations of stochastic gradient descent")
    ylabel("Estimate of expected conditional log probability density");
    title("$latent net scores ")
    plt.savefig(string(save_loc,"scores.png"))
    plt.close()
    
end

function plot_hist_vs_1Dprior(proposal, latent, vartoplot, gp_val, t_val, source_params; save_loc="./")

    h=[]; n=length(gp_val)
    for i = 1:1000
        max_scene_duration = 2.5; min_scene_duration = 0.5;
        scene_duration = round((max_scene_duration-min_scene_duration)*rand() + min_scene_duration,digits=3)
        choices,_,=propose(proposal,(Dict(latent=>gp_val,:t=>t_val),scene_duration,))
        push!(h,choices[latent => vartoplot])
    end
    hist(h,bins=20,density=true)
    sp = source_params["gp"][string(latent)][string(vartoplot)]
    xvals = vartoplot == :mu ? (0:50) : 0:0.05:6
    plot(xvals,exp.([Gen.logpdf(sp["dist"],x,sp["args"]...) for x in xvals]))
    legend(["pdf","samples"])
    title("Hist $vartoplot with $n gp point vs. prior pdf")
    plt.savefig(string(save_loc, string(latent), "_", string(vartoplot) ,"__hist_vs_prior.png"))
    plt.close()

end

function plot_correspondence(proposal, data_generator, latent, vartoplot, n_points, n_same; save_loc="./")
    
    actual = []
    predicted = []
    for z = 1:n_points
        i,c = data_generator()
        push!(actual, c[latent=>vartoplot])
        m = []
        for j = 1:n_same
            trace, _, _ = propose(proposal,i)
            push!(m, trace[latent=>vartoplot])
        end
        push!(predicted, mean(m))
    end
    scatter(actual,predicted,marker=".")
    xlabel("Actual")
    ylabel("Mean of $n_same proposal samples")
    r = round(cor(actual,predicted),digits = 4)
    title("$latent $vartoplot, r=$r, n=$n_points")
    maxy = maximum(predicted); miny=minimum(predicted);
    maxx = maximum(actual); minx =minimum(actual);
    xlim([min(miny,minx)-0.5, max(maxy,maxx)+0.5])
    ylim([min(miny,minx)-0.5, max(maxy,maxx)+0.5])
    plt.savefig(string(save_loc, string(latent), "_", string(vartoplot) ,"__correspondence.png"))
    plt.close()

end

function modtrain!(gen_fn::GenerativeFunction, data_generator::Function,
                update::ParamUpdate;
                num_epoch=1, epoch_size=1, num_minibatch=1, minibatch_size=1,
                evaluation_size=epoch_size, eval_inputs_and_constraints=[], verbose=false)

    history = Vector{Float64}(undef, num_epoch)
    
    if length(eval_inputs_and_constraints) == 0 
        println("Making evaluation dataset:")
        for i = 1:evaluation_size
            print("$i ")
            ic = data_generator()
            push!(eval_inputs_and_constraints, ic)
        end
        println()
    else
        evaluation_size = length(eval_inputs_and_constraints)
        println("Using eval set of size $evaluation_size")
    end
    
    for epoch=1:num_epoch

        s1=time()
        # generate data for epoch
        if verbose
            println("epoch $epoch: generating $epoch_size training examples...")
        end
        epoch_inputs = Vector{Tuple}(undef, epoch_size)
        epoch_choice_maps = Vector{ChoiceMap}(undef, epoch_size)
        for i=1:epoch_size
            (epoch_inputs[i], epoch_choice_maps[i]) = data_generator()
        end
        s2=time()-s1

        # train on epoch data
        s1=time()
        if verbose
            println("Time to generate data: $s2")
            println("epoch $epoch: training using $num_minibatch minibatches of size $minibatch_size...")
        end
        for minibatch=1:num_minibatch
            permuted = Random.randperm(epoch_size)
            minibatch_idx = permuted[1:minibatch_size]
            minibatch_inputs = epoch_inputs[minibatch_idx]
            minibatch_choice_maps = epoch_choice_maps[minibatch_idx]
            for (inputs, constraints) in zip(minibatch_inputs, minibatch_choice_maps)
                (trace, _) = generate(gen_fn, inputs, constraints)
                retval_grad = accepts_output_grad(gen_fn) ? zero(get_retval(trace)) : nothing
                accumulate_param_gradients!(trace, retval_grad)
            end
            apply!(update)
        end
        s2=time()-s1
        
        # evaluate score on held out data
        s1=time()
        if verbose
            println("Time to train: $s2")
            println("epoch $epoch: evaluating on $evaluation_size examples...")
        end
        avg_score = 0.
        for i=1:evaluation_size
            (_, weight) = generate(gen_fn, eval_inputs_and_constraints[i][1], eval_inputs_and_constraints[i][2])
            avg_score += weight
        end
        avg_score /= evaluation_size
        @assert ~isnan(avg_score)
        history[epoch] = avg_score
        s2=time()-s1

        if verbose
            println("Time to evaluate: $s2")
            println("epoch $epoch: est. objective value: $avg_score")
        end
        
        

    end
    return history
end
