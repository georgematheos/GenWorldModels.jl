function make_constraints(sentences::AbstractVector{Tuple{Int, Int, Int}}, num_entities)
    constraints = choicemap(
        (:world => :num_relations => () => :num, NUM_REL_PRIOR_MEAN),
    )
    for e1=1:num_entities, e2=1:num_entities, r=1:NUM_REL_PRIOR_MEAN
        constraints[:world => :num_facts => (Relation(r), Entity(e1), Entity(e2)) => :is_true] = true
    end
    # true_for_rel = [Set() for _=1:NUM_REL_PRIOR_MEAN]
    # for r=1:NUM_REL_PRIOR_MEAN
    #     constriants[:world => :sparsity => Relation(r) => :sparsity] = 0.2
    #     certainly_true1 = uniform_discrete(1, num_entities)
    #     certainly_true2 = uniform_discrete(1, num_entities)
    #     push!(true_for_rel
    # end
    for (i, sentence) in enumerate(sentences)
        constraints[:kernel => :rels_and_sentences => i => :verb => :verb] = sentence[2]
        rel = uniform_discrete(1, NUM_REL_PRIOR_MEAN)
        constraints[:kernel => :facts => :sampled_facts => i] = Fact((Relation(rel), Entity(sentence[1]), Entity(sentence[3])), 1)
    end
    return constraints
end

function get_initial_trace(sentences::AbstractVector{Tuple{Int, Int, Int}}, num_entities, num_verbs)
    constraints = make_constraints(sentences, num_entities)
    tr, _ = generate(generate_sentences, (num_entities, num_verbs, length(sentences)), constraints; check_proper_usage=false)
    return tr
end