@type Cluster

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