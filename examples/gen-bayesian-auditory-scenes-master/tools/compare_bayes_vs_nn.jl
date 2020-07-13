using PyCall
using GenTF
tf = pyimport("tensorflow")
#2020-02-10 04:39:58.942004: E tensorflow/stream_executor/cuda/cuda_blas.cc:654] failed to run cuBLAS routine cublasSgemmBatched: CUBLAS_STATUS_EXECUTION_FAILED
#https://stackoverflow.com/questions/38303974/tensorflow-running-error-with-cublas
config = tf.ConfigProto()
config.gpu_options.allow_growth = true
session = tf.Session(config=config)

include("./tf17_custom_proposals.jl")

using GaussianProcesses
using Gen;
using Random;
using Optim
using JSON; using Printf

using Statistics: mean, std, cor;
using LinearAlgebra: dot;
using StatsFuns: logsumexp, softplus;
using PyPlot
using SpecialFunctions: digamma,trigamma;
include("./time_helpers.jl")
include("./extra_distributions.jl")
include("./gaussian_helpers.jl")

println("Setting random seed...")
random_seed = abs(rand(Int,1)[1])
Random.seed!(random_seed)

latent = :erb
basepath="/om2/user/mcusi/gen-bayesian-auditory-scenes/proposals/erb_73/"
checkpoint_name="40-nn.ckpt.ckpt"
batch_size = 50; 
evaluation_size = 2;

##Basic parameters & priors 
source_params, steps, gtg_params, obs_noise = include("./base_params.jl")
audio_sr = 20000;
source_latent_model = make_source_latent_model(source_params, audio_sr, steps);

latents = Dict()
latents[:wait] = Dict(:tp => :wait)
latents[:dur_minus_min] = Dict(:tp => :dur_minus_min)
latents[:amp1D] = Dict(:gp => :amp, :source_type => "tone")
latents[:amp2D] = Dict(:gp => :amp, :source_type => "noise")
latents[:erb] = Dict(:gp => :erb, :source_type => "tone")

data_generators = Dict(); evalsets=Dict(); 
if batch_size <= 1 
	data_generators[latent] = make_data_generator(source_latent_model, latents[latent])
elseif batch_size > 1
	data_generators[latent] = make_batch_data_generator(source_latent_model, latents[latent], batch_size)
end
evalsets[latent] = evaluation_size > 1 ? [data_generators[latent]() for i = 1:evaluation_size] : [data_generators[latent]()]

trainable_proposals = Dict(); weights = Dict(); net_funcs = Dict(); 
if (latent == :erb || latent == :amp) && batch_size <= 1
	# net_funcs[latent], weights[latent] = define_neural_net(5, 1, 16, 8, dilations=[1,2,4,8]);
	# trainable_proposals[latent] = create_trainable_gp1D_proposal(latent, net_funcs[latent]);
	net_funcs[latent], weights[latent] = define_neural_net(5, 1, 16, 8, 1, dilations=[1,2,4,8]);
	trainable_proposals[latent] = create_trainable_gp1D_proposal(latent, net_funcs[latent]);
else (latent == :erb || latent == :amp ) && batch_size > 1
	net_funcs[latent], weights[latent] = define_neural_net(5, 1, 24, 19, batch_size, dilations=[1,2,4,8,16]);
	trainable_proposals[latent] = create_trainable_batch_gp1D_proposal(latent, net_funcs[latent]);	
end


svs = [:mu, :scale, :sigma, :epsilon]

## Storage dictionaries
MLE = Dict(); HMC = Dict(); NN = Dict(); C = Dict();
for v in svs
	MLE[v]=[]; HMC[v]=Dict(:mean=>[],:std=>[],:samples=>[]); 
	C[v]=[]; NN[v]=Dict(:mean=>[],:std=>[],:samples=>[]); 
end

##Necessary for HMC evaluation
hyperpriors = latent == :erb ? source_params["gp"][string(latent)] : 
	(source_type == "noise" || source_type == "harmonic" ? source_params["gp"][string(latent)]["2D"] : source_params["gp"][string(latent)]["1D"])
prior_samples = Dict()
for v in svs 
	hyperprior = hyperpriors[string(v)]
	prior_samples[v] = Dict(:samples=>[],:mean=>0.0,:std=>0.0)
	if v == :mu
		append!(prior_samples[v][:samples],[ hyperprior["dist"](hyperprior["args"]...) for g in 1:1000 ])
	else
		append!(prior_samples[v][:samples], [ log( hyperprior["dist"](hyperprior["args"]...) ) for g in 1:1000 ])
    end
    prior_samples[v][:mean] = mean(prior_samples[v][:samples])
    prior_samples[v][:std] = std(prior_samples[v][:samples])
end

##Load in the neural network
vardict = Dict([w.name => w for w in weights[latent]]...) 
ckpt_machine = tf.train.Saver(vardict)
ckpt_machine.restore(GenTF.get_session(net_funcs[latent]), 
			string(basepath,checkpoint_name))

d = evalsets[latent]
for i = 1:evaluation_size

	##Neural network
	#1. prepare neural network variables 
	scene_duration = d[i][1][2]
	scene_t = get_element_gp_times([0,scene_duration], steps["t"]) #If training on variable length scenes, change this
	net_inputs = []; masks = []; datapoints_ms = []; datapoints_ss = [];

	for j = 1:batch_size

		datapoint = d[i][1][1][j]

		## Ground Truth 
		for v in svs
			push!(C[v], d[i][2][(latent, j)=>v])
		end

		## MLE
		n_mle = 3; gps = []; scores=[];
		for k = 1:n_mle

			mConstant = GaussianProcesses.MeanConst( Gen.uniform(hyperpriors["mu"]["args"]...) )
	        kern = GaussianProcesses.SE(rand(prior_samples[:scale][:samples]), rand(prior_samples[:sigma][:samples]))
	        logObsNoise = rand(prior_samples[:epsilon][:samples])
	        gp = GaussianProcesses.GP(Float64.(datapoint[:t]),
	        	Float64.(datapoint[:erb]),mConstant, kern, logObsNoise)
	         
	        optimize!(gp; domean=true, kern=true, noise=true, 
	        		meanbounds=[[hyperpriors["mu"]["args"][1]],[hyperpriors["mu"]["args"][2]]], 
	        		kernbounds = [[-15, -15], [5, 5]])
	        
	        push!(scores, gp.mll) #marginal log likelihood
	        push!(gps, gp) 

		end
		gp = gps[argmax(scores)]
		push!(MLE[:mu], GaussianProcesses.get_params(gp.mean)[1])
    	kernel_params = GaussianProcesses.get_params(gp.kernel)
    	push!(MLE[:scale], exp(kernel_params[1]))
    	push!(MLE[:sigma], exp(kernel_params[2]))
    	push!(MLE[:epsilon], exp(GaussianProcesses.get_params(gp.logNoise)[1]))

		## HMC 
		mean_func = GaussianProcesses.MeanConst(rand(prior_samples[:mu][:samples]))
		kern = GaussianProcesses.SE(rand(prior_samples[:scale][:samples]), rand(prior_samples[:sigma][:samples]))
		log_obs_noise = rand(prior_samples[:epsilon][:samples])
		gp_ess = GaussianProcesses.GP(Float64.(datapoint[:t]), Float64.(datapoint[latent]), mean_func, kern, log_obs_noise)

		set_priors!(gp_ess.mean, [Distributions.Normal(prior_samples[:mu][:mean], prior_samples[:mu][:std])]) #I don't know why this could be Uniform in the notebook and not here??
		set_priors!(gp_ess.kernel, [Distributions.Normal(prior_samples[:scale][:mean], prior_samples[:scale][:std]),
										Distributions.Normal(prior_samples[:sigma][:mean], prior_samples[:sigma][:std])]) 
		set_priors!(gp_ess.logNoise, [Distributions.Normal(prior_samples[:epsilon][:mean], prior_samples[:epsilon][:std])])
		@time chain = ess(gp_ess, nIter=5000)

		ess_svs = [:epsilon, :mu, :scale, :sigma] #order arbitrarily used by GaussianProcesses pkg
		for k = 1:length(ess_svs)
			l = ess_svs[k]
			if l == :epsilon || l == :scale || l == :sigma
		        ess_samples = exp.(chain[k,:])
		    else
		        ess_samples = chain[k,:]
		    end
		    push!(HMC[l][:mean], mean(ess_samples))
		    push!(HMC[l][:std], std(ess_samples))
		    push!(HMC[l][:samples], rand(ess_samples))
		end

		##Neural network -- 2. format inputs 
        mask = [t in datapoint[:t] ? 1.0 : 0.0 for t in scene_t]
        embedded_gp = []; k =1 
        datapoint_mean = mean(datapoint[latent]); 
        for t_idx in 1:length(scene_t)
            if mask[t_idx] == 1
                push!(embedded_gp, datapoint[latent][k])
                k = k + 1
            else
                push!(embedded_gp, datapoint_mean)
            end
        end
        datapoint_std = length(datapoint[latent]) > 1 ? std(datapoint[latent]) : 0.0
        s_div = length(datapoint[latent]) > 1 ? datapoint_std : 1.0
        embedded_gp = (embedded_gp .- datapoint_mean)./s_div

        net_input = cat(embedded_gp, mask, dims=2)
        push!(net_inputs, reshape(net_input, 1, size(net_input)...))
        mask = cat(mask...,dims=1)
        push!(masks, reshape(mask, 1, size(mask)...))
        push!(datapoints_ms, datapoint_mean); push!(datapoints_ss, datapoint_std)

	end

	#Neural Net: 3. Batch formatting 
    batch_net_input = cat(net_inputs..., dims=1)
    batch_mask_input = cat(masks..., dims=1)
    batch_m = batch_size > 1 ? cat(datapoints_ms..., dims=1) : [datapoints_ms[1]]
    batch_s = batch_size > 1 ? cat(datapoints_ss..., dims=1) : [datapoints_ss[1]]

    in_channels=2; in_height=length(scene_t); in_width=1;
    batch_net_input = Float32.(reshape(batch_net_input, batch_size, in_height, in_width, in_channels))
    batch_mask_input = Float32.(reshape(batch_mask_input, batch_size, in_height, 1, 1))
    batch_m = Float32.( (reshape(batch_m, batch_size, 1) .- 20.0)./20.0 )
    batch_s = Float32.( (reshape(batch_s, batch_size, 1) .- 2.0)./2.0 )

    #4. run through network 
	(net_trace,_) = generate(net_funcs[latent], (batch_net_input,batch_mask_input, batch_m, batch_s,))
	dparams = get_retval(net_trace)

    for j = 1:batch_size

	    mean_mu_estimate = 20.0*dparams[j,1] .+ 20 ; 
	    shape_mu = softplus(2*dparams[j,2])
	    q_mu = normal(mean_mu_estimate, shape_mu)
	    push!(NN[:mu][:mean], mean_mu_estimate)
	    push!(NN[:mu][:std], shape_mu)
	    push!(NN[:mu][:samples], q_mu)
	          
	    mean_sigma_estimate = softplus( (2.0*dparams[j,3] + 2.0)  + dparams[j,4]*q_mu/20.0 )
	    shape_sigma = softplus(dparams[j,5] + dparams[j,6]*q_mu/20.0)
	    q_sigma = gamma(shape_sigma, mean_sigma_estimate/shape_sigma)
	    sigma_std = sqrt( shape_sigma * (mean_sigma_estimate/shape_sigma)^2 )
	  	push!(NN[:sigma][:mean], mean_sigma_estimate)
	    push!(NN[:sigma][:std], sigma_std )
	    push!(NN[:sigma][:samples], q_sigma)

	    mean_scale_estimate = softplus(dparams[j,7]  + dparams[j,8]*q_sigma); #+ dparams[j,8]*q_mu
	    shape_scale = softplus(dparams[j,9] + dparams[j,11]*q_sigma); #+ dparams[j,10]*q_mu
	    q_scale = gamma(shape_scale, mean_scale_estimate/shape_scale)
	  	scale_std = sqrt( shape_scale * (mean_scale_estimate/shape_scale)^2 )
	  	push!(NN[:scale][:mean], mean_scale_estimate)
	    push!(NN[:scale][:std], scale_std )
	    push!(NN[:scale][:samples], q_scale)

	    mean_epsilon_estimate = softplus(dparams[j,12] + dparams[j,14]*q_sigma + dparams[j,15]*q_scale); #+ dparams[j,13]*q_mu
	    shape_epsilon = softplus(dparams[j,16] + dparams[j,18]*q_sigma + dparams[j,19]*q_scale);  #+ dparams[i,17]*q_mu
	    q_epsilon = gamma(shape_epsilon, mean_epsilon_estimate/shape_epsilon)
	  	epsilon_std = sqrt( shape_epsilon * (mean_epsilon_estimate/shape_epsilon)^2 )
	  	push!(NN[:epsilon][:mean], mean_epsilon_estimate)
	    push!(NN[:epsilon][:std], epsilon_std )
	    push!(NN[:epsilon][:samples], q_epsilon)

	end

end

##Plotting 
function plotinfo(x,y)
    r = round(cor(x,y),digits = 4)
    maxy = maximum(y); miny=minimum(y);
    maxx = maximum(x); minx =minimum(x);
    limits=[min(miny,minx)-0.5, max(maxy,maxx)+0.5]
    return limits, r
end

## Compare Point-estimates 
for v in svs 
	point_estimate_list = [C[v], MLE[v], HMC[v][:samples], HMC[v][:mean], NN[v][:samples], NN[v][:mean]]
	point_estimate_names = ["Actual", "MLE", "HMCsample", "HMCmean", "NNsample", "NNmean"]
	for i_x = 1:length(point_estimate_list)
		x = point_estimate_list[i_x]
		xn = point_estimate_names[i_x]
		for i_y = i_x+1:length(point_estimate_list)
			y = point_estimate_list[i_y]
			yn = point_estimate_names[i_y]
			if (   (xn == "HMCsample" && yn == "HMCmean") #not very meaningful
				|| (xn == "HMCsample" && yn == "NNmean") #better version below
				|| (xn == "HMCmean" && yn == "NNsample")  #better version below
				|| (xn == "HMCmean" && yn == "NNmean") #doing exactly this below
				|| (xn == "NNsample" && yn == "NNmean")) #not very meaningful
				continue
			end
			scatter(x, y, marker=".")
			limits, r = plotinfo(x, y)
			xlim(limits); ylim(limits)
			title("$latent $v: $xn vs. $yn, r=$r")
			xlabel(xn); ylabel(yn)
			plt.savefig(string(basepath, xn, "_", yn, "_$v","_cor.png"))
			plt.close()
		end
	end
end

## Neural network vs. HMC, mean and std 
for summary_stat in [:mean, :std]
	for v in svs 
		scatter(HMC[v][summary_stat], NN[v][summary_stat], marker=".")
		limits, r = plotinfo(HMC[v][summary_stat], NN[v][summary_stat])
		title("$latent $v: HMC vs. NN $summary_stat, r=$r")
		xlabel("HMC $summary_stat"); ylabel("NN $summary_stat")
		plt.savefig(string(basepath, "HMC_NN_$v","_$summary_stat.png"))
		plt.close()
	end
end

println("complete!")

