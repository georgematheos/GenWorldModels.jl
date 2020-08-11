@gen function sdds_proposal(tr)
    do_smart_split_dumb_merge ~ bernoulli(0.5)
    if do_smart_split_dumb_merge
        {*} ~ smart_split_dumb_merge_proposal(tr)
    else
        {*} ~ smart_split_dumb_merge_proposal(tr)
    end
end

@gen function smart_split_dumb_merge_proposal(tr)
    do_split ~ bernoulli(0.5)
    if do_split
        {*} ~ smart_split_proposal(tr)
    else
        {*} ~ dumb_merge_proposal(tr)
    end
end

@gen function smart_split_proposal(tr)
    (counts, indicess_per_entity) = tr[:kernel => :counts]
    num_objs = tr[:world => :num_objects => ()]
    probs = [exp(logbeta(α + counts[i]) - logbeta(counts[i])) for i=1:num_objs]
    to_split ~ categorical(probs)

    new_idx1 ~ uniform_discrete(1, num_objs)
    new_idx2 ~ uniform_discrete(new_idx1, num_objs + 1)

    total_num_1 = 0
    num_1_per_men = zeros(NUM_WORDS)
    total_num_2 = 0
    num_2_per_men = zeros(NUM_WORDS)
    tot_α = sum(α)
    for idx in indices_per_entity[to_split]
        mention = tr[:kernel => :sampled_mentions][idx]
        p1 = (α_val + num_1_per_men[mention])/(tot_α + tot_num_1)
        p2 = (α_val + num_2_per_men[mention])/(tot_α + tot_num_2)
        to_idx_1 = {:to_idx_1 => idx} ~ bernoulli(p1 / (p1 + p2))

        if to_idx1
            total_num_1 += 1
            num_1_per_men[mention] += 1
        else
            total_num_2 += 1
            num_2_per_men[mention] += 1
        end
    end

    dirichlet1 ~ dirichlet(num_1_per_men)
    dirichlet2 ~ dirichlet(num_2_per_men)
end

@oupm_involution sdds_inv (old, fwd) to (new, bwd) begin
    do_split = @read(fwd[:do_split], :disc)
    if do_split
        from_idx = @read(fwd[:to_split], :disc)
        to_idx1, to_idx2 = @read(fwd[:to_idx1], :disc), @read(fwd[:to_idx2], :disc)
        @split(Object(from_idx), to_idx1, to_idx2)
        
        (_, indices_per_entity) = @read(old[:counts], :disc)
        for idx in indices_per_entity[from_idx]
            to_1 = @read(fwd[:to_idx_1 => idx], :disc)
            @write(new[:sampled_objs => idx => :mention], to_1 ? to_idx1 : to_idx2, disc)
        end
    else

    end
end