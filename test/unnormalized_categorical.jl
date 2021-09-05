@type Clust
@oupm generate_clusters_and_samples() begin
    @number Clust() = (return num ~ poisson_plus_1(5))
    @property unnormalized_weight(::Clust) ~ gamma(5.0, 1.0)
    @observation_model (static, diffs) function get_values()
        cluster_to_itself = lazy_set_to_dict_map(identity, @objects(Clust))
        weights = @dictmap (c => @get(unnormalized_weight[clust]) for (c, clust) in cluster_to_itself)

        # Now actually do the sampling:
        cluster_samples ~ unnormalized_categorical(@world(), 5, weights)
        return cluster_samples
    end
end
@load_generated_functions()

tr, generate_weight = generate(generate_clusters_and_samples, (), choicemap(
    @set_number(Clust(), 5),
    (@obsmodel() => :cluster_samples => 1, Clust(1)),
    (@obsmodel() => :cluster_samples => 2, Clust(1)),
    (@obsmodel() => :cluster_samples => 3, Clust(1)),
    (@obsmodel() => :cluster_samples => 4, Clust(1)),
    (@obsmodel() => :cluster_samples => 5, Clust(2))
));

new_tr, update_weight, retdiff, discard = update(tr, (), (), choicemap(
    (@obsmodel() => :cluster_samples => 4, Clust(2))
))

probs = [@get(tr, unnormalized_weight[Clust(i)]) for i=1:5] |> x -> x./sum(x)
old_sampled_prob = 4*log(probs[1]) + log(probs[2])
new_sampled_prob = 3*log(probs[1]) + 2*log(probs[2])
@test isapprox(update_weight, new_sampled_prob - old_sampled_prob)
@test retdiff == VectorDiff(5, 5, Dict(4 => UnknownChange()))

new_tr, update_weight, retdiff, discard = update(tr, (), (), choicemap(
    @set(unnormalized_weight[Clust(1)], 10.),
    @set(unnormalized_weight[Clust(2)], 4.),
))
new_probs = vcat([10., 4.], [@get(new_tr, unnormalized_weight[Clust(i)]) for i=3:5]) |> x -> x./sum(x)
old_sampled_prob = 4*log(probs[1]) + log(probs[2])
new_sampled_prob = 4*log(new_probs[1]) + log(new_probs[2])
change_from_gammas = logpdf(gamma, 10., 5., 1.) + logpdf(gamma, 4., 5., 1.) - (
    logpdf(gamma, @get(tr, unnormalized_weight[Clust(1)]), 5., 1.) + logpdf(gamma, @get(tr, unnormalized_weight[Clust(2)]), 5., 1.)
)
@test isapprox(update_weight, new_sampled_prob - old_sampled_prob + change_from_gammas)

new_tr, update_weight, retdiff, discard = update(tr, (), (), choicemap(
    @set_number(Clust(), 6),
    @set(unnormalized_weight[Clust(6)], 2.0)
))
new_probs = [@get(new_tr, unnormalized_weight[Clust(i)]) for i=1:6] |> x -> x./sum(x)
old_sampled_prob = 4*log(probs[1]) + log(probs[2])
new_sampled_prob = 4*log(new_probs[1]) + log(new_probs[2])
change_from_num = logpdf(poisson_plus_1, 6, 5) - logpdf(poisson_plus_1, 5, 5)
@test isapprox(update_weight, new_sampled_prob - old_sampled_prob + change_from_num + logpdf(gamma, 2., 5., 1.))

new_new_tr, update_weight, retdiff, discard = update(new_tr, (), (), choicemap(
    (@obsmodel() => :cluster_samples => 4, Clust(2))
))
old_sampled_prob = 4*log(new_probs[1]) + log(new_probs[2])
new_sampled_prob = 3*log(new_probs[1]) + 2*log(new_probs[2])
@test isapprox(update_weight, new_sampled_prob - old_sampled_prob)
@test retdiff == VectorDiff(5, 5, Dict(4 => UnknownChange()))

new_new_tr, update_weight, retdiff, discard = update(new_tr, (), (), choicemap(
    (@obsmodel() => :cluster_samples => 3, Clust(2))
))
new_new_sampled_prob = 2*log(new_probs[1]) + 3*log(new_probs[2])
@test isapprox(update_weight, new_new_sampled_prob - new_sampled_prob)
@test retdiff == VectorDiff(5, 5, Dict(3 => UnknownChange()))
