function run_ancestral_sampling!(initial_tr, num_iters, examine!; examine_freq=1)
    run_inference!(initial_tr, ancestral_inference_iter, num_iters, examine!; examine_freq)
end

function ancestral_inference_iter(tr)
    tr = reassoc_relation(tr)
    tr = update_truth_value(tr)
    tr = change_num_rels(tr)
    return tr
end

@gen function reassoc_relation_proposal(tr, sentence_idx)
    entpair = entpairs(tr)[sentence_idx]
    facts = [Fact(r, entpair...) for r=1:tr[:num_rels]]
    fact = {:sampled_facts => :sampled_facts => sentence_idx} ~ uniform_from_list(facts)
end
function reassoc_relation(tr)
    idx = uniform_discrete(1, length(get_retval(tr)[1]))
    tr, acc = mh(tr, reassoc_relation_proposal, (idx,))
    return tr
end

@gen function change_truth_value_proposal(tr, rel, ent1, ent2)
    (α, β) = beta_prior(tr)
    prob = α/(α + β)
    {:sampled_facts => :all_facts => :facts_per_rel => rel => :true_entpairs => (ent1, ent2)} ~ bernoulli(prob)
end
function update_truth_value(tr)
    rel = uniform_discrete(1, tr[:num_rels])
    ent1 = uniform_discrete(1, num_ents(tr))
    ent2 = uniform_discrete(1, num_ents(tr))
    tr, acc = mh(tr, change_truth_value_proposal, (rel, ent1, ent2))
    return tr
end

function change_num_rels(tr)
    tr, acc = mh(tr, select(:num_rels))
    return tr
end