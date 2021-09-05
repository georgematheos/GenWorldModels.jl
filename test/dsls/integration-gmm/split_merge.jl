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
    w1, w2 = @get(tr, unnormalized_weight[to_merge1] => :wt, unnormalized_weight[to_merge2] => :wt)
    mean1, var1 = @get(tr, mean[to_merge1] => :mean, var[to_merge1] => :var)
    mean2, var2 = @get(tr, mean[to_merge2] => :mean, var[to_merge2] => :var)
    
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

    constraints[@addr(mean[new1] => :mean)] = mean1
    constraints[@addr(var[new1] => :var)] = var1
    constraints[@addr(mean[new2] => :mean)] = mean2
    constraints[@addr(var[new2] => :var)] = var2
    
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
