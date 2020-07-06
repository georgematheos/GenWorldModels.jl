module BallsModel
using Gen

include("../../src/WorldModels.jl")
using .WorldModels
import .WorldModels: @w, @WorldMap, @UsingWorld

function getarg(world, symbol)
    return 5
end

#########
# Model #
#########

@dist color(world::WorldModels.World, ball) = uniform_discrete(1, getarg(world, :num_colors))

@gen (static) function ball(world, sample)
    num_balls ~ @w num_balls[()]
    return {ball} ~ uniform_discrete(1, num_balls)
end

@gen function obs_color(world, sample)
    ball ~ @w ball[sample]
    if {:correct_obs} ~ bernoulli(0.9)
        return {:color} ~ @w color[ball]
    else
        return {:color} ~ uniform_discrete(1, world[:num_colors])
    end
end

@gen (static) function _sample_colors(world, num_samples)
    return {:samples} ~ @WorldMap(:obs_color, 1:num_samples)
end

sample_colors = @UsingWorld(_sample_colors, obs_color, ball, color; world_args=[:num_colors])

#############
# Inference #
#############
observations = choicemap(
        (:samples => 1, 4),
        (:samples => 2, 10),
        (:samples => 3, 1),
        (:samples => 4, 2),
        (:samples => 5, 5),
        (:samples => 6, 7)
)
model_args = (10, 6)

(trs, log_norm_weights, lml_est) = importance_sampling(
    sample_colors, model_args, observations, 10000, true
)

tr = generate(sample_colors, model_args, observations)
for i=1:10000
    type = uniform_discrete(1, 3)
    if type == 1 # change draw
        idx = uniform_discrete(1, model_args[2])
        tr, _ = mh(tr, select(:world => :ball => idx))
    elseif type == 2 # change color
        idx = uniform_discrete(1, tr[:world => :num_balls => ()])
        tr, _ = mh(tr, select(:world => :color => idx))
    elseif type == 3 # change observed color
        idx = uniform_discrete(tr, model_args[1])
        tr, _ = mh(tr, select(:world => :obs_color => idx))
    end

    if bernoulli(0.2)
        tr, _ = mh(tr, select(:world => :num_balls => ()))
    end
end

end