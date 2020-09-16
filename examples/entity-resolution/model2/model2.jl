@type Entity; @type Relation; @type Fact;

@dist num_relations(world, t) = log_normal(NUM_RELS_MEAN)

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
    verbs ~ integrated_dirichlet_to_categorical(rels, α)

    (verbs, entpairs)
end