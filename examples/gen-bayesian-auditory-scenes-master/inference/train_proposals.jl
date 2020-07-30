using Gen;
using JSON; 
using Printf;
include("../inference/custom_proposals.jl")
println("Loaded packages...")

latent = :erb
#[:wait, :dur_minus_min, :amp1D, :amp2D, :erb]

println("Setting random seed...")
random_seed = abs(rand(Int,1)[1])
Random.seed!(random_seed)

searchdir(path,key) = filter(x->occursin(key,x), readdir(path))
list_of_expts = searchdir("./proposals/",string(latent))
iter = length(list_of_expts) + 1
itername = @sprintf("%02d",iter); 
save_loc = string("./proposals/",string(latent),"_$itername/");
print("Making folder: "); println(save_loc)
mkdir(save_loc)
d = Dict()
d["seed"] = random_seed
open(string(save_loc, "random_seed.json"),"w") do f
    JSON.print(f, d)
end

##Basic parameters & priors 
source_params, steps, gtg_params, obs_noise = include("./base_params.jl")
audio_sr = 20000;
source_latent_model = make_source_latent_model(source_params, audio_sr, steps);
batch_size = 256; 

println("Making data generators")
latents = Dict()
latents[:wait] = Dict(:tp => :wait)
latents[:dur_minus_min] = Dict(:tp => :dur_minus_min)
latents[:amp1D] = Dict(:gp => :amp, :source_type => "tone")
latents[:amp2D] = Dict(:gp => :amp, :source_type => "noise")
latents[:erb] = Dict(:gp => :erb, :source_type => "tone")

data_generators = Dict(); evalsets=Dict(); evaluation_size = 3;
if batch_size <= 1 
	data_generators[latent] = make_data_generator(source_latent_model, latents[latent])
elseif batch_size > 1
	data_generators[latent] = make_batch_data_generator(source_latent_model, latents[latent], batch_size)
end
println("Generating $evaluation_size eval set...")
evalsets[latent] = [data_generators[latent]() for i = 1:evaluation_size];
println("Done generating $evaluation_size eval set")

println("Defining proposals")
trainable_proposals = Dict(); weights = Dict()
if latent == :wait || latent == :dur_minus_min
	trainable_proposals[latent] = create_trainable_tp_proposal(:wait)
	weights[latent] = [:B_mean_mu,:B_shape_mu,:B_mean_alpha,:B_shape_a,
			:MLE_estimate_alpha_0,:MOM_estimate_alpha_0,
		    :C_shape_mu, :C_shape_a]
end
net_funcs = Dict(); 
if (latent == :erb || latent == :amp) && batch_size <= 1
	# net_funcs[latent], weights[latent] = define_neural_net(5, 1, 16, 8, dilations=[1,2,4,8]);
	# trainable_proposals[latent] = create_trainable_gp1D_proposal(latent, net_funcs[latent]);
	net_funcs[latent], weights[latent] = define_neural_net(5, 1, 16, 8, 1, dilations=[1,2,4,8]);
	trainable_proposals[latent] = create_trainable_gp1D_proposal(latent, net_funcs[latent]);
else (latent == :erb || latent == :amp ) && batch_size > 1
	net_funcs[latent], weights[latent] = define_neural_net(5, 1, 24, 19, batch_size, dilations=[1,2,4,8,16]);
	trainable_proposals[latent] = create_trainable_batch_gp1D_proposal(latent, net_funcs[latent]);	
end

if latent == :wait || latent == :dur_minus_min

	println("Initializing parameters")
	Gen.init_param!(trainable_proposals[latent], :B_mean_mu, [1.0, 0.0])
	Gen.init_param!(trainable_proposals[latent], :B_mean_alpha, zeros(3))
	for p in [:MLE_estimate_alpha_0, :MOM_estimate_alpha_0, 
	        :B_shape_mu, :B_shape_a, :C_shape_mu, :C_shape_a]
	    Gen.init_param!(trainable_proposals[latent], p, 0.0)
	end

	update = Gen.ParamUpdate(Gen.GradientDescent(1e-4,100), 
		trainable_proposals[latent]);
	n_loops = 10

	println("starting training")
	for loop = 1:n_loops
		scores = modtrain!(trainable_proposals[latent], 
			data_generators[latent], 
			update,
		    num_epoch=10, epoch_size=2000, num_minibatch=500, 
		    minibatch_size=10, evaluation_size=100, 
		    eval_inputs_and_constraints=evalsets[latent], 
		    verbose=false);	
		loop_loc = string(save_loc,"$loop-")
		plot_scores(latent, scores; save_loc=loop_loc)
		trained_weights = [ string(w) => Gen.get_param(trainable_proposals[latent], w) for w in weights[latent] ]
		trained_weights_dict = Dict(trained_weights...)
		open(string(loop_loc, "weights.json"),"w") do f
			JSON.print(f, trained_weights_dict)
	    end
	    for sv in [:mu, :a]
		    plot_correspondence(trainable_proposals[latent], data_generators[latent], 
		    	latent, sv, 100, 1; save_loc=loop_loc)
		    # plot_hist_vs_tpprior(trainable_proposals[latent], latent, sv, 
		    # 	latent == :wait ? [0.12,0.11] : [0.11], 
		    # 	source_params; save_loc=loop_loc)
	    end
	end

elseif latent == :erb || latent == :amp1D 

	println("initializing update")
	update = Gen.ParamUpdate(Gen.FixedStepGradientDescent(1e-10),
	     net_funcs[latent] => collect(Gen.get_params(net_funcs[latent])));
	n_loops = 50 #determines how often you save weights & print summary images

	println("starting training")
	saver = get_net_saver(net_funcs[latent])
	all_scores = []
	for loop = 1:n_loops
		println("Loop $loop")
		scores = modtrain!(trainable_proposals[latent], 
		    data_generators[latent], 
		    update,
		    num_epoch=10, epoch_size=100, num_minibatch=1, 
		    minibatch_size=100, evaluation_size=2, 
		    eval_inputs_and_constraints=evalsets[latent], 
		    verbose=true);	
		loop_loc = string(save_loc,"$loop-")
		save_net_weights(saver, net_funcs[latent], "nn.ckpt"; save_loc=loop_loc)
		append!(all_scores, scores)
		plot_scores(latent, all_scores; save_loc=loop_loc)
		for sv = [:mu, :sigma, :scale, :epsilon]
			if batch_size <= 1 
				plot_hist_vs_1Dprior(trainable_proposals[latent], latent, sv, [20.0], [0.22], source_params, 100; save_loc=loop_loc)
				plot_correspondence(trainable_proposals[latent], data_generators[latent], latent, sv, 100, 1; save_loc=loop_loc)
			else
				plot_hist_vs_1Dprior_batch(trainable_proposals[latent], latent, sv, [20.0], [0.22], source_params, 2, batch_size; save_loc=loop_loc)
				#correspondence_batch will plot 2*batch_size points
				plot_correspondence_batch(trainable_proposals[latent], data_generators[latent], latent, sv, 2, batch_size; save_loc=loop_loc)
			end
		end

	end

end

println("Complete")
