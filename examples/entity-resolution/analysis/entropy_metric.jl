using StatsBase

function get_rel_to_sentence_indices(st::State)
    rel_to_sentence_indices = [Set() for i=1:st.num_relations]
    for (i, rel) in enumerate(st.sentence_rels)
        push!(rel_to_sentence_indices[rel], i)
    end
    return rel_to_sentence_indices
end

function entropy_of_second_given_first(first::State, second::State)
    rel_to_first_indices = get_rel_to_sentence_indices(first)
    first_freqs = map(length, rel_to_first_indices)
    first_freqs = first_freqs ./ sum(first_freqs)
    total_ent = 0
    for (freq, first_indices) in zip(first_freqs, rel_to_first_indices)
        second_rels = [second.sentence_rels[idx] for idx in first_indices]
        second_rel_freqs = [length(findall(x->x===r, second_rels)) for r in unique(second_rels)]
        second_rel_freqs = second_rel_freqs ./ sum(second_rel_freqs)
        total_ent += freq * entropy(second_rel_freqs)
    end
    return total_ent
end