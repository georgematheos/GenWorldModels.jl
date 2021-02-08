@type Cluster
@dist poisson_plus_1(λ) = poisson(λ) + 1
@oupm gaussian_mixture_model(λ, ξ, κ, α_v, β_v, α_w, β_w) begin
    @number Cluster() = (return :num ~ poisson_plus_1(@arg λ))
    @property (static, diffs) function mean(::Cluster)
        return :mean ~ normal(@arg ξ, 1/(@arg κ))
    end
    @property (static, diffs) function var(::Cluster)
        return :var ~ normal(@arg α_v, @arg β_v)
    end
    @property (static, diffs) function unnormalized_weight(::Cluster)
        return :wt ~ gamma(@arg α_w, @arg β_w)
    end

    @property (static, diffs) function cluster_to_weight()
        cluster_to_itself = lazy_set_to_dict_map(identity, @objects(Cluster))
        return @dictmap (
            c => @get(unnormalized_weight[clust])
            for (c, clust) in cluster_to_itself
        )
    end
    @observation_model (static, diffs) function get_values(num_datapoints)
        cluster_samples ~ unnormalized_categorical(@world(), num_datapoints, @get(cluster_to_weight[]))
        means = @map [mean(c) for c in cluster_samples]
        vars = @map [var(c) for c in cluster_samples]
        return :vals ~ Map(normal)(means, vars)
    end
end

@kernel function split_merge_kernel(tr)
    prob_merge = @get_number(tr, Cluster()) <: 1 ? 0.0 : 0.5
    do_merge ~ bernoulli(prob_merge)
    if do_merge
        return {*} ~ merge_kernel(tr)
    else
        return {*} ~ split_kernel(tr)
    end
end

@kernel function merge_kernel(tr)
    clusters = @objects(tr, Cluster())
    to_merge1 ~ uniform_choice(clusters)
    to_merge2 ~ uniform_from_list([c for c in clusters if c != to_merge1])
    new_idx ~ uniform_discrete(1, length(clusters) - 1)
    new_cluster = Cluster(new_idx)

    mean1, var1, w1 = @get(tr, mean[to_merge1] => :mean, var[to_merge1] => :var, unnormalized_weight[to_merge1] => :wt)
    mean2, var2, w2 = @get(tr, mean[to_merge2] => :mean, var[to_merge1] = :var, unnormalized_weight[to_merge2] => :wt)

    wt, mean, var = merged_component_params(mean1, var1, w1, mean2, var2, w2)
    u1, u2, u3 = reverse_split_params(mean, var, wt, mean1, var1, w1, mean2, var2, w2)

    return (
        WorldUpdate!(tr,
            Merge(new_cluster, merge1, merge2),
            choicemap(
                @set mean[new_cluster] = mean,
                @set var[new_cluster] = var,
                @set unnormalized_weight[new_cluster] = wt
            )
        ),
        choicemap(
            (:do_merge, false), (:to_split, new_cluster),
            (:idx1, @index(tr, to_merge1)), (:idx2, @index(tr, to_merge2)),
            (:u1, u1), (:u2, u2), (:u3, u3)
        )
    )
end
@kernel function split_kernel(tr)
    clusters = @objects(tr, Cluster())
    to_split ~ uniform_choice(clusters)
    idx1 ~ uniform_discrete(1, length(clusters) + 1)
    idx2 ~ uniform_from_list([i for i=1:(length(clusters) + 1) if i != idx1])
    new1, new2 = Cluster(idx1), Cluster(idx2)

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
        @set mean[new1] = mean1, @set var[new1] = var1, @set unnormalized_weight[new1] = u1*old_wt,
        @set mean[new2] = mean2, @set var[new2] = var2, @set unnormalized_weight[new2] = (1-u1)*old_wt
    )

    for i in tr[@obsmodel() => :cluster_samples => :indices_for_cluster][@abstract(tr, to_split)]
        y = tr[@obsmodel() => :vals => i]
        p = u1 * exp(logpdf(normal, y, mean1, var1)),
        q = (1 - u1) * exp(logpdf(normal, y, mean2, var2))
        if {:to_first => i} ~ bernoulli(p/(p+q))
            constraints[@obsmodel() => :cluster_samples => i] = new1
        else
            constraints[@obsmodel() => :cluster_samples => i] = new2
        end
    end

    return (
        WorldUpdate!(tr,
            Split(to_split, idx1, idx2),
            constraints
        ),
        choicemap(
            (:do_merge, true), (:to_merge1, new1), (:to_merge2, new2),
            (:new_idx, @index(tr, to_split))
        )
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

@testset begin "Integration test: Richardson & Green Split/Merge for Gaussian Mixture Model"
    # TODO
end