include("distributions.jl")
include("dirichlet_process_entity_mention.jl")
include("beta_bernoulli_subset.jl")

@type Relation
struct Fact
    rel::Relation
    ent1::Int
    ent2::Int
end

@gen (static, diffs) function num_relations(world, t)
    num_rels_prior ~ lookup_or_generate(world[:args][:num_rels_prior])
    num ~ discrete_log_normal(num_rels_prior[1], num_rels_prior[2])
    return num
end

fact_getter(rel) = entpair -> Fact(rel, entpair...)
fact_getter(rel::Diffed{Int, NoChange}) = Diffed(fact_getter(strip_diff(rel)), NoChange())
@gen (static, diffs) function generate_facts_for_rel(args)
    (rel, possible_entpairs, beta_prior) = args
    true_entpairs ~ beta_bernoulli_subset(possible_entpairs, beta_prior[1], beta_prior[2])
    facts = no_collision_set_map(fact_getter(rel), true_entpairs)
    return facts
end

@gen (static, diffs) function get_fact_set(world)
    num_entities ~ lookup_or_generate(world[:args][:num_entities])
    possible_entpairs = [(i, j) for i=1:num_entities, j=1:num_entities]
    beta_prior ~ lookup_or_generate(world[:args][:beta_prior])

    rels ~ get_sibling_set(:Relation, :num_relations, world, ())

    rel_to_args_dict = lazy_set_to_dict_map(rel -> (rel, possible_entpairs, beta_prior), rels)
    rels_to_facts ~ DictMap(generate_facts_for_rel)(rel_to_args_dict)
    facts ~ tracked_union(rels_to_facts)
    return facts
end

@gen (static, diffs) function sample_facts(world, num_sentences)
    facts ~ get_fact_set(world)
    sampled_facts ~ Map(uniform_choice)(fill(facts, num_sentences))
    return sampled_facts
end

@gen (static, diffs) function _generate_sentences(world, num_sentences, dirichlet_prior_val, num_verbs)
    facts ~ sample_facts(world, num_sentences)
    rels = map(fact -> fact.rel, facts)
    entpairs = map(fact -> (fact.ent1, fact.ent2), facts)
    
    α = fill(dirichlet_prior_val, num_verbs)
    verbs ~ dirichlet_process_entity_mention(rels, α)

    return (verbs, entpairs)
end

@load_generated_functions()

"""
    generate_sentences(num_entities, num_rels_prior, beta_prior, num_sentences, dirichlet_prior_val, num_verbs)

`beta_prior = (α, β)`, `num_rels_prior = (μ, σ)` (for the lognormal, so the mean is exp(μ + σ^2/2))
"""
generate_sentences = UsingWorld(_generate_sentences, :num_relations => num_relations;
    world_args=(:num_entities, :num_rels_prior, :beta_prior)
)