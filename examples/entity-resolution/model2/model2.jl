include("distributions.jl")
include("dirichlet_process_entity_mention.jl")
include("beta_bernoulli_subset.jl")

@type Relation
struct Fact
    rel::Relation
    ent1::Int
    ent2::Int
end

@dist num_relations(world, t) = log_normal(NUM_RELS_MEAN)

fact_getter(rel) = entpair -> Fact(rel, entpair...)
fact_getter(rel::Diffed{Int, NoChange}) = Diffed(fact_getter(strip_diff(rel)), NoChange())
@gen (static, diffs) function generate_facts_for_rel(num_entities, rel, α, β)
    possible_entpairs = [(i, j) for i=1:num_entities, j=1:num_entities]
    true_entpairs ~ beta_bernoulli_subset(possible_entpairs, α, β)
    facts = no_collision_set_map(fact_getter(rel), true_entpairs)
    return facts
end

@gen (static, diffs) function get_fact_set(world)
    rels ~ get_sibling_set(:Relation, :num_relations, world, ())
    rels_to_facts ~ DictMap(GenerateFactsForRel(NUM_ENTITIES))(rels)
    facts ~ tracked_union(rels_to_facts)
    return facts
end

@gen (static, diffs) function sample_facts(world, num_sentences)
    facts ~ get_fact_set(world)
    sampled_facts ~ Map(uniform_choice)(fill(world, num_sentences), fill(facts, num_sentences))
    return sampled_facts
end

@gen (static, diffs) function _generate_sentences(world, num_sentences, dirichlet_prior_val, num_verbs)
    facts ~ sample_facts(world, num_sentences)
    origins = Map(lookup_or_generate)(mgfcall_map(world[:origin], facts))
    rels = map(((rel, e1, e2),) -> rel, origins)
    entpairs = map(((rel, e1, e2),) -> (e1, e2), origins)
    
    α = fill(dirichlet_prior_val, num_verbs)
    verbs ~ dirichlet_process_entity_mention(rels, α)

    (verbs, entpairs)
end

generate_sentences = UsingWorld(_generate_sentences, :num_relations => num_relations)