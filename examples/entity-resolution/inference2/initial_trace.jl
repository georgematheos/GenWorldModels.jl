true_entpairs_addr(rel) = :kernel => :sampled_facts => :all_facts => :rels_to_facts => rel => :true_entpairs
true_entpairs_addr(rel, suff) = :kernel => :sampled_facts => :all_facts => :rels_to_facts => rel => :true_entpairs => suff

function make_constraints(sentences::Vector{SentenceNumeric}, params::ModelParams)
    constraints = choicemap()
    num_rels = discrete_log_normal(params.num_relations_prior...)
    constraints[:world => :num_relations => () => :num] = num_rels

    true_for_rel = [Set() for _=1:num_rels]
    for (i, s) in enumerate(sentences)
        rel = uniform_discrete(1, num_rels)
        push!(true_for_rel[rel], (s.ent1, s.ent2))
        constraints[:kernel => :sampled_facts => :sampled_facts => i] = Fact(Relation(rel), s.ent1, s.ent2)
        constraints[:kernel => :verbs => i] = s.verb
    end

    for (r, certainly_true) in enumerate(true_for_rel)
        # for ep in certainly_true
        #     constraints[true_entpairs_addr(Relation(r), ep)] = true
        # end
        constraints[true_entpairs_addr(Relation(r))] = certainly_true
    end

    return constraints
end

function get_initial_trace(sentences::Vector{SentenceNumeric}, params::ModelParams)
    constraints = make_constraints(sentences, params)
    tr, _ = generate(generate_sentences, model_args(params), constraints; check_proper_usage=false)
    return tr
end