struct Evaluation
    total_ent_given_gt
    total_ent_given_inferred
end


# ground_truth_fact_to_indices = [(fact_idx, Set(indices...)), ...]
# idx_to_ground_truth_fact[idx] = fact_idx

function evaluate(inferred_tr, groundtruth_tr)
    
end

function evaluate(tr, ground_truth_fact_to_idx, idx_to_ground_truth_fact)
    total_ent_given_gt = get_entropy_of_other(ground_truth_fact_to_idx, idx -> tr[:kernel => :facts => :sampled_facts => idx])

    fact_to_sentences = Dict()
    for idx in 1:length(get_retval(tr))
        fact = tr[:kernel => :facts => :sampled_facts => idx]
        fact_to_sentences[fact] = push!(get(fact_to_sentences, fact, Set()), idx)
    end
    total_ent_given_inferred = get_entropy_of_other(ground_truth, idx -> idx_to_ground_truth_fact[idx])

    return Evaluation(total_ent_given_gt, total_ent_given_inferred)
end

function get_entropy_of_other(given, get_other_fact_for_index)
    entr = 0.
    for (fact, sentence_indices) in given
        inferred_facts = Set([get_other_fact_for_index(idx) for idx in sentence_indices])
        inferred_fact_counts = Dict()
        for fct in inferred_facts
            inferred_fact_counts[fct] = get(inferred_fact_counts, fct, 0) + 1
        end
        countvec = collect(values(inferred_fact_counts))
        probvec = countvec/sum(countvec)
        entr += StatsBase.entropy(Categorical(probvec))
    end 
    return entr
end