function make_constraints(sentences::Vector{SentenceNumeric}, params::ModelParams)
    constraints = choicemap()
    num_rels = discrete_log_normal(params.num_relations_prior...)
    constraints[:num_rels] = num_rels

    true_for_rel = [Set() for _=1:num_rels]
    for (i, s) in enumerate(sentences)
        rel = uniform_discrete(1, num_rels)
        push!(true_for_rel[rel], (s.ent1, s.ent2))
        constraints[:sampled_facts => :sampled_facts => i] = Fact(rel, s.ent1, s.ent2)
    end

    for (r, certainly_true) in enumerate(true_for_rel)
        constraints[:sampled_facts => :all_facts => :facts_per_rel => r => :true_entpairs] = certainly_true
    end

    return constraints
end

function get_initial_trace(sentences::Vector{SentenceNumeric}, params::ModelParams)
    constraints = make_constraints(sentences, params)
    tr, _ = generate(generate_sentences, model_args(params), constraints)
    return tr
end