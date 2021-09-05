# Here we pay the price for using the gamma + normalize rather than using dirichlet!
# We need extra auxiliary variables to account for the sum of the normalizations.
# Should still be a gibbs update.  Not sure the computational complexity is worse this way,
# but it's certainly uglier!

# TODO: currently this is not always being accepted!
# I'm not sure what is going wrong...I worked through the math pretty carefully and was pretty sure
# the acceptance ratio worked out to 1.
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
        p = [exp(logpdf(normal, y, mean, var) + log(wt)) for (mean, var, wt) in zip(means, vars, wts)]
        probs = p ./ sum(p) .+ 1e-10 # add 1e-10 to each slot to avoid floating-point error due to extremely small log-scale gaussian probabilities
        sample = {@obsmodel() => :cluster_samples => i} ~ categorical_from_list([Cluster(i) for i=1:nc], probs)
    end
end