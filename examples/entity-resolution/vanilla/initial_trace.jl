function make_constraints(sentences::Vector{SentenceNumeric}, params::ModelParams)
    constraints = choicemap()
    num_rels = discrete_log_normal(params.num_relations_prior...)
    constraints[:num_rels] = num_rels

    true_for_rel = [Set() for _=1:num_rels]
    for (i, s) in enumerate(sentences)
        rel = uniform_discrete(1, num_rels)
        push!(true_for_rel[rel], (s.ent1, s.ent2))
        constraints[:sampled_facts => :sampled_facts => i] = Fact(rel, s.ent1, s.ent2)
        constraints[:verbs => i] = s.verb
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

function make_constraints_for_state(state, sentences, params)
    constraints = choicemap()
    constraints[:num_rels] = state.num_relations
    for (i, (rel, sentence)) in enumerate(zip(state.sentence_rels, sentences))
        constraints[:sampled_facts => :sampled_facts => i] = Fact(rel, sentence.ent1, sentence.ent2)
        constraints[:verbs => i] = sentence.verb
    end
    entpair_sets = [Set() for _=1:state.num_relations]
    for fact in state.facts
        push!(entpair_sets[fact.rel], (fact.ent1, fact.ent2))
    end
    for (i, set) in enumerate(entpair_sets)
        constraints[:sampled_facts => :all_facts => :facts_per_rel => i => :true_entpairs] = set
    end

    return constraints
end
function get_trace_for_state(state, sentences, params)
    constraints = make_constraints_for_state(state, sentences, params)
    tr, _ = generate(generate_sentences, model_args(params), constraints)
    return tr
end