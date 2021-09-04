using Revise
using Gen
using GenWorldModels
using Test

### Model ###
@type Cluster
@dist poisson_plus_1(λ) = poisson(λ) + 1

# To simulate sampling from a dirichlet(δ), set α_w = δ; β_w = 1

@oupm gaussian_mixture_model(λ, ξ, κ, α_v, β_v, α_w, β_w) begin
    @number Cluster() = (return num ~ poisson_plus_1(@arg λ))
    @property (static, diffs) function mean(::Cluster)
        return mean ~ normal((@arg ξ), 1/(@arg κ))
    end
    @property (static, diffs) function var(::Cluster)
        return var ~ inv_gamma((@arg α_v), (@arg β_v))
    end
    @property (static, diffs) function unnormalized_weight(::Cluster)
        return wt ~ gamma((@arg α_w), (@arg β_w))
    end

    @observation_model (static, diffs) function get_values(num_datapoints)
        cluster_to_itself = lazy_set_to_dict_map(identity, @objects(Cluster))
        weights = @dictmap (c => @get(unnormalized_weight[clust]) for (c, clust) in cluster_to_itself)

        # I don't want to use MSSIs in the inference algorithm:
        # I want all the properties to be generated for each object.
        all_means = @dictmap (c => @get(mean[clust]) for (c, clust) in cluster_to_itself)
        all_vars = @dictmap (c => @get(var[clust]) for (c, clust) in cluster_to_itself)

        # Now actually do the sampling:
        cluster_samples ~ unnormalized_categorical(@world(), num_datapoints, weights)
        means = @map [@get(mean[c]) for c in cluster_samples]
        vars = @map [@get(var[c]) for c in cluster_samples]
        return vals ~ Map(normal)(means, vars)
    end
end
@load_generated_functions()

### Split/Merge Kernel ###
@kernel function split_merge_kernel(tr)
    prob_merge = @get_number(tr, Cluster()) <= 1 ? 0.0 : 0.5
    do_merge ~ bernoulli(prob_merge)
    do_merge ? {*} ~ merge_kernel(tr) : {*} ~ split_kernel(tr)
end

@kernel function merge_kernel(tr)
    clusters = @objects(tr, Cluster())
    to_merge1 ~ uniform_choice(clusters)
    to_merge2 ~ uniform_from_list([c for c in clusters if c != to_merge1])
    new_idx ~ uniform_discrete(1, length(clusters) - 1)
    new_cluster = Cluster(new_idx)
    
    indices = tr[@obsmodel() => :cluster_samples => :obj_to_indices]
    # any_associated_with_1 = length(indices[@abstract(tr, to_merge1)]) > 0
    # any_associated_with_2 = length(indices[@abstract(tr, to_merge2)]) > 0

    w1, w2 = @get(tr, unnormalized_weight[to_merge1] => :wt, unnormalized_weight[to_merge2] => :wt)

    # if !(any_associated_with_1 || any_associated_with_2)
    #     # if there are no associations, this is easy
    #     println("short circuiting for merge")

    #     return (
    #         WorldUpdate!(tr, Merge(new_cluster, to_merge1, to_merge2)),
    #         choicemap((:do_merge, false), (:to_split, new_cluster),
    #         (:idx1, @index(tr, to_merge1)), (:idx2, @index(tr, to_merge2)))
    #     )
    # end

    # if any_associated_with_1
        mean1, var1 = @get(tr, mean[to_merge1] => :mean, var[to_merge1] => :var)
    # else
    #     mean1 ~ normal(@arg(tr, ξ), 1/@arg(tr, κ))
    #     var1 ~ inv_gamma(@arg(tr, α_v), @arg(tr, β_v))
    # end
    # if any_associated_with_2
        mean2, var2 = @get(tr, mean[to_merge2] => :mean, var[to_merge2] => :var)
    # else
    #     mean2 ~ normal(@arg(tr, ξ), 1/@arg(tr, κ))
    #     var2 ~ inv_gamma(@arg(tr, α_v), @arg(tr, β_v))
    # end

    wt, mean, var = merged_component_params(mean1, var1, w1, mean2, var2, w2)
    u1, u2, u3 = reverse_split_params(mean, var, wt, mean1, var1, w1, mean2, var2, w2)

    constraints = choicemap(
            @set(mean[new_cluster] => :mean, mean),
            @set(var[new_cluster] => :var, var),
            @set(unnormalized_weight[new_cluster] => :wt, wt),
            ( # reassociate with the new cluster
                (@obsmodel() => :cluster_samples => i, new_cluster)
                for i in Iterators.flatten((indices[@abstract(tr, to_merge1)], indices[@abstract(tr, to_merge2)]))
            )...
        )

    return (
        WorldUpdate!(tr,
            Merge(new_cluster, to_merge1, to_merge2),
            constraints
        ),
        choicemap(
            (:do_merge, false), (:to_split, new_cluster),
            (:idx1, @index(tr, to_merge1)), (:idx2, @index(tr, to_merge2)),
            (:u1, u1), (:u2, u2), (:u3, u3),
            ( # set reverse direction assignments
                (:to_first => i, true)
                for i in indices[@abstract(tr, to_merge1)]
            )...,
            (
                (:to_first => i, false)
                for i in indices[@abstract(tr, to_merge2)]
            )...
        )
    )
end
@kernel function split_kernel(tr)
    clusters = @objects(tr, Cluster())
    to_split ~ uniform_choice(clusters)
    idx1 ~ uniform_discrete(1, length(clusters) + 1)
    idx2 ~ uniform_from_list([i for i=1:(length(clusters) + 1) if i != idx1])
    new1, new2 = Cluster(idx1), Cluster(idx2)
    
    # this gives a very strange error message if we incorrectly do `[to_split]` rather than `[@abstract(tr, to_split)]`;
    # we should improve the error message behavior
    # if length(tr[@obsmodel() => :cluster_samples => :obj_to_indices][@abstract(tr, to_split)]) == 0
    #     # if there are no associations we can skip most of the work
    #     println("short circuiting for split")
    #     return (
    #         WorldUpdate!(tr, Split(to_split, idx1, idx2)),
    #         choicemap((:do_merge, true), (:to_merge1, new1),
    #             (:to_merge2, new2), (:new_idx, @index(tr, to_split)))
    #     )
    # end

    u1 ~ beta(2, 2)
    u2 ~ beta(2, 2)
    u3 ~ beta(1, 1)
    mean, var = @get(tr, mean[to_split], var[to_split])
    mean1 = mean - u2*sqrt(var * (1 - u1)/ u1)
    mean2 = mean + u2*sqrt(var + u1/(1 - u1))
    var1 = u3*(1 - u2^2)*var/u1
    var2 = (1 - u3)*(1 - u2^2)*var/(1 - u1)
    old_wt = @get(tr, unnormalized_weight[to_split])

    constraints = choicemap(
        @set(unnormalized_weight[new1] => :wt, u1*old_wt),
        @set(unnormalized_weight[new2] => :wt, (1-u1)*old_wt)
    )
    bwd_move = choicemap(
        (:do_merge, true), (:to_merge1, new1), (:to_merge2, new2),
        (:new_idx, @index(tr, to_split))
    )

    num_for_1 = 0
    num_for_2 = 0
    for i in tr[@obsmodel() => :cluster_samples => :obj_to_indices][@abstract(tr, to_split)]
        y = tr[@obsmodel() => :vals => i]
        p = u1 * exp(logpdf(normal, y, mean1, var1))
        q = (1 - u1) * exp(logpdf(normal, y, mean2, var2))
        if ({:to_first => i} ~ bernoulli(p/(p+q)))
            constraints[@obsmodel() => :cluster_samples => i] = new1
            num_for_1 += 1
        else
            constraints[@obsmodel() => :cluster_samples => i] = new2
            num_for_2 += 1
        end
    end

    # if num_for_1 > 0
        constraints[@addr(mean[new1] => :mean)] = mean1
        constraints[@addr(var[new1] => :var)] = var1
    # else
    #     bwd_move[:mean1] = mean1
    #     bwd_move[:var1] = var1
    # end
    # if num_for_2 > 0
        constraints[@addr(mean[new2] => :mean)] = mean2
        constraints[@addr(var[new2] => :var)] = var2
    # else
    #     bwd_move[:mean2] = mean2
    #     bwd_move[:var2] = var2
    # end
    
    return (
        WorldUpdate!(tr,
            Split(to_split, idx1, idx2),
            constraints
        ), bwd_move
    )
end

function merged_component_params(mean1, var1, w1, mean2, var2, w2)
    w = w1 + w2
    mean = (w1*mean1 + w2*mean2) / w
    var = -mean^2 + (w1*(mean1^2 + var1) + w2*(mean2^2 + var2)) / w
    return (w, mean, var)
end

function reverse_split_params(mean, var, w, mean1, var1, w1, mean2, var2, w2)
    u1 = w1/w
    u2 = (mean - mean1) / sqrt(var * w2/w1)
    u3 = var1/var * u1 / (1 - u2^2)
    return (u1, u2, u3)
end

### Gibbs updates ###

# Here we pay the price for using the gamma + normalize rather than using dirichlet!
# We need extra auxiliary variables to account for the sum of the normalizations.
# Should still be a gibbs update.  Not sure the computational complexity is worse this way,
# but it's certainly uglier!

# TODO: currently this is not always being accepted!  I'm not sure what is going wrong...I think this
# is a gibbs move.
@kernel function update_w(tr)
    indices = tr[@obsmodel() => :cluster_samples => :obj_to_indices]
    vals = Dict()
    cnt = 0
    for cluster in @objects(tr, Cluster)
        cnt += @arg(tr, α_w) + length(indices[@abstract(tr, cluster)])
        vals[cluster] = {:unnorm => cluster} ~ gamma(
           @arg(tr, α_w) + length(indices[@abstract(tr, cluster)]), @arg(tr, β_w)
        )
    end
    
    base_alpha_sum = @arg(tr, α_w) * @get_number(tr, Cluster())
    post_normalization_sum ~ gamma(base_alpha_sum, @arg(tr, β_w))
    # = sum of weights in new world

    count_sum_in_current_world = sum(length(indices[@abstract(tr, cluster)]) for cluster in @objects(tr, Cluster()))
    @assert base_alpha_sum + count_sum_in_current_world == cnt
    bwd_normalization_sum ~ gamma(base_alpha_sum + count_sum_in_current_world, @arg(tr, β_w))
    # = during the reverse move, when we sample from the gammas, what will the sum of that be

    current_sum_from_gammas = sum(values(vals))
    scalar = post_normalization_sum / current_sum_from_gammas

    current_sum_in_world = sum(@get(tr, unnormalized_weight[cluster] => :wt) for cluster in @objects(tr, Cluster()))
    bwd_scalar = bwd_normalization_sum / current_sum_in_world

    # println("sum in new world: post_normalization_sum = $post_normalization_sum")
    # println("current sum in world: current_sum_in_world = $current_sum_in_world")
    # println("current_sum_from_gammas = $current_sum_from_gammas")
    # println("bwd_normalization_sum = $bwd_normalization_sum")
    # println("base_alpha_sum: $base_alpha_sum")

    return (
        choicemap(
            ((@addr(unnormalized_weight[cluster] => :wt), scalar * val)
            for (cluster, val) in vals)...
        ),
        choicemap(
            (:post_normalization_sum, current_sum_in_world),
            (:bwd_normalization_sum, current_sum_from_gammas),
            (
                (:unnorm => cluster, @get(tr, unnormalized_weight[cluster] => :wt) * bwd_scalar)
                for cluster in @objects(tr, Cluster())
            )...
        )
    )
end

@gen function update_means(tr)
    κ, ξ = @arg(tr, κ), @arg(tr, ξ)
    for cluster in @objects(tr, Cluster)
        vals = [get_retval(tr)[i] for i in tr[@obsmodel() => :cluster_samples => :obj_to_indices][cluster]]
        if !isempty(vals)
            n, μ, σ² = length(vals), @get(tr, mean[cluster]), @get(tr, var[cluster])
            {@addr(mean[cluster] => :mean)} ~ normal(
                (sum(vals)/σ² + κ * ξ)/(n/σ² + κ),
                1/(n/σ² + κ)
            )
        end
    end
end
@gen function update_vars(tr)
    α, β = @arg(tr, α_v), @arg(tr, β_v)
    for cluster in @objects(tr, Cluster)
        vals = [get_retval(tr)[i] for i in tr[@obsmodel() => :cluster_samples => :obj_to_indices][cluster]]
        if !isempty(vals)
            n, μ = length(vals), @get(tr, mean[cluster])
            {@addr(vars[cluster] => :var)} ~ inv_gamma(
                α + n/2, β + sum((vals .- μ).^2)/2
            )
        end
    end
end

@dist categorical_from_list(list, probs) = list[categorical(probs)]
@gen function update_allocations(tr)
    n_vals = get_args(tr)[end]
    nc = @get_number(tr, Cluster())
    means = [@get(tr, mean[Cluster(i)] => :mean) for i=1:nc]
    vars = [@get(tr, var[Cluster(i)] => :var) for i=1:nc]
    wts = [@get(tr, unnormalized_weight[Cluster(i)] => :wt) for i=1:nc]
    for i=1:n_vals
        y = tr[@obsmodel() => :vals => i]
        p = [exp(logpdf(normal, y, mean, var))*wt for (mean, var, wt) in zip(means, vars, wts)]
        {@obsmodel() => :cluster_samples => i} ~ categorical_from_list([Cluster(i) for i=1:nc], p ./ sum(p))
    end
end

### Inference loop ###
split_merge(tr) = mh(tr, MHProposal(split_merge_kernel); check=false)[1]# check=true, roundtrip_atol=0.5)[1]
function w_move(tr)
    a = false
    for _=1:4
        tr, acc = mh(tr, MHProposal(update_w); check=false)
        a = a || acc
    end
    if !a
        @warn "w move not accepted"
    end
    # @assert acc "w was not gibbs"
    return tr
end
gibbs(proposal, update_name) = tr ->
    let (newtr, acc) = mh(tr, proposal, ())
        if !acc
            @warn "move for $update_name was rejected (should be gibbs)"
        end
        newtr
    end

function inference_cycle(tr)
    tr = w_move(tr)
    tr = gibbs(update_means, "means")(tr)
    tr = gibbs(update_vars, "vars")(tr)
    tr = gibbs(update_allocations, "allocations")(tr)
    tr = split_merge(tr)
    return tr
end
function do_inference(tr, n_iters; get_map=false)
    map = tr
    for i=1:n_iters
        tr = inference_cycle(tr)
        map = get_score(map) > get_score(tr) ? map : tr
    end
    return get_map ? map : tr
end

### Run inference ###

# Params from https://github.com/mugamma/gmm/blob/master/tests/three_component_mixture.jl
# λ, ξ, κ, α_v, β_v = 3, 0.0, 0.01, 2.0, 10.0
# δ = 5.0 (parameter for weight_vec ~ dirichlet(δ))
params = (3, 0.0, 0.01, 2.0, 10.0, 5.0, 1.0)

### Split/merge unit test ###
# function trace_for_which_we_expect_a_split()
#     cluster1 = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3]
#     cluster2 = [5.0, 5.1, 5.2, 5.05]
#     cluster3 = [-3.0, -2.5, -2.8, -2.9, -2.68]
#     vals = [cluster1..., cluster2..., cluster3...]
#     tr, _ = generate(
#         gaussian_mixture_model,
#         (params..., length(vals)),
#         choicemap(
#             @set_number(Cluster(), 2),
#             @set(mean[Cluster(1)] => :mean, 0.0), @set(mean[Cluster(2)] => :mean, 5.0),
#             @set(var[Cluster(1)] => :var, 1.0), @set(var[Cluster(2)] => :var, 1.0),
#             @set(unnormalized_weight[Cluster(1)] => :wt, 1.0), @set(unnormalized_weight[Cluster(2)] => :wt, 1.0),
#             (
#                 (
#                     @obsmodel() => :cluster_samples => i,
#                     i > length(cluster1) ? Cluster(2) : Cluster(1)
#                 )
#                 for i=1:length(vals)
#             )...,
#             ((@obsmodel() => :vals => i, val) for (i, val) in enumerate(vals))...
#         )
#     )
#     return tr
# end

# @testset "gets_split" begin
#     tr = trace_for_which_we_expect_a_split()
#     any_accepted = false
#     for i=1:10
#         tr, acc = mh(tr, MHProposal(split_merge_kernel))
#         any_accepted = any_accepted || acc
#     end
#     println()
#     @test any_accepted
#     @test @get_number(tr, Cluster()) == 3
# end

### Successful Inference Test ###
# Testset adapted from
# https://github.com/mugamma/gmm/blob/master/tests/three_component_mixture.jl
value_constraints(vals) = choicemap(((@obsmodel() => :vals => i, val) for (i, val) in enumerate(vals))...)
# @testset "correctly infers three component mixture" begin
    ys = [11.26, 28.93, 30.52, 30.09, 29.46, 10.03, 11.24, 11.55, 30.4, -18.44,
          10.91, 11.89, -20.64, 30.59, 14.84, 13.54, 7.25, 12.83, 11.86, 29.95,
          29.47, -18.16, -19.28, -18.87, 9.95, 28.24, 9.43, 7.38, 29.46, 30.73,
          7.75, 28.29, -21.99, -20.0, -20.86, 15.5, -18.62, 13.11, 28.66,
          28.18, -18.78, -20.48, 9.18, -20.12, 10.2, 30.26, -14.94, 5.45, 31.1,
          30.01, 10.52, 30.48, -20.37, -19.3, -21.92, -18.31, -18.9, -20.03,
          29.32, -17.53, 10.61, 6.38, -20.72, 10.29, 11.21, -18.98, 8.57,
          10.47, -22.4, 6.58, 29.8, -17.43, 7.8, 9.72, -21.53, 11.76, 29.72,
          29.31, 6.82, 15.51, 10.69, 29.56, 8.84, 30.93, 28.75, 10.72, 9.21,
          8.57, 11.92, -23.96, -19.78, -17.2, 11.79, 29.95, 7.29, 6.57, -17.99,
          13.29, -22.53, -20.0]
    zs = [2, 3, 3, 3, 3, 2, 2, 2, 3, 1, 2, 2, 1, 3, 2, 2, 2, 2, 2, 3, 3, 1, 1,
          1, 2, 3, 2, 2, 3, 3, 2, 3, 1, 1, 1, 2, 1, 2, 3, 3, 1, 1, 2, 1, 2, 3,
          1, 2, 3, 3, 2, 3, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1,
          2, 3, 1, 2, 2, 1, 2, 3, 3, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 2, 1, 1, 1,
          2, 3, 2, 2, 1, 2, 1, 1]
    @assert length(ys) == length(zs)

    initial_tr, weight = generate(
        gaussian_mixture_model,
        (params..., length(ys)),
        value_constraints(ys)
    );
    @test get_score(initial_tr) > -Inf
    println("Initial trace generation successful.")

    inferred_tr = do_inference(initial_tr, 100; get_map = false)
    @test @get_number(inferred_tr, Cluster()) == 3

    true_means = [-20., 10., 30.]
    true_vars = [3.0, 5.0, 1.0]

    mean_error(i, mean) = abs(@get(inferred_tr, mean[Cluster(i)]) - mean)
    closest_idx_to_mean(mean) = argmin([mean_error(i, mean) for i=1:3])
    idxs = [closest_idx_to_mean(mean) for mean in true_means]

    for (idx, mean, var) in zip(idxs, true_means, true_vars)
        @test mean_error(idx, mean) < 1.0
        @test abs(@get(inferred_tr, var[Cluster(idx)]) - var) < 2.0
    end

    inferred_idx(inferred_tr, test_to_world_idx, datapoint_idx) = findfirst([Cluster(i) == inferred_tr[@obsmodel() => :cluster_samples => datapoint_idx] for i in test_to_world_idx])
    n_misallocated = sum([zs[datapoint_idx] != inferred_idx(inferred_tr, idxs, datapoint_idx) for datapoint_idx=1:length(ys)])
    println("$n_misallocated / $(length(zs)) samples misallocated")
    @test n_misallocated < 0.05 * length(ys)
# end
#=
TODOs:
1. [x] Make sure that the changes to the update algorithm are correct.  Make sure I understand
the invariant of when we need to call `note_new_call!` and when `has_val` becomes true.
2. [ ] Get `update_w` to be a Gibbs update!  [But I should be careful not to spend more than another hour or two
on the GMM example not doing GWM debugging.]
=#