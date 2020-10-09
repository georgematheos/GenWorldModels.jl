### SPLIT PROPOSAL ###
@gen function split_proposal(tr, is_smart)
    current_num_rels = tr[:num_rels]
    if is_smart
        from_idx = {*} ~ sample_smartsplit_relation(tr, current_num_rels)
    else
        from_idx ~ uniform_discrete(1, current_num_rels)
    end

    new_idx1 ~ uniform_discrete(1, current_num_rels + 1)
    new_idx2 ~ uniform_from_list([idx for idx=1:(current_num_rels + 1) if idx !== new_idx1])
    {*} ~ sample_split_assignments(tr, from_idx, is_smart)
end

@gen function sample_smartsplit_relation(tr, current_num_rels)
    function cnt(rel)
        if haskey(tr[:verbs => :counts], rel)
            tr[:verbs => :counts][rel]
        else
            zeros(num_verbs(tr))
        end
    end
    logfactors = [
        -logbeta(cnt(rel) .+ dirichlet_prior_val(tr))
        for rel=1:current_num_rels
    ]
    logprobs = logfactors .- logsumexp(logfactors)
    probs = exp.(logprobs)
    return {:from_idx} ~ categorical(probs)
end

@gen function sample_split_assignments(tr, from_idx, is_smart)
    count1 = zeros(Int, num_verbs(tr))
    count2 = zeros(Int, num_verbs(tr))
    itr = haskey(tr[:verbs => :indices_per_entity], from_idx) ? tr[:verbs => :indices_per_entity][from_idx] : ()
    for idx in itr
        mention = verbs(tr)[idx]
        if is_smart
            p1 = (dirichlet_prior_val(tr) + count1[mention]) / (num_verbs(tr) * dirichlet_prior_val(tr) + sum(count1))
            p2 = (dirichlet_prior_val(tr) + count2[mention]) / (num_verbs(tr) * dirichlet_prior_val(tr) + sum(count2))
            # println("about to sample for :to_first => $idx")
            to_first = {:to_first => idx} ~ bernoulli(p1/(p1+p2))
        else
            # println("about to sample for :to_first => $idx")
            to_first = {:to_first => idx} ~ bernoulli(0.5)
        end
        if to_first
            count1[mention] += 1
        else
            count2[mention] += 1
        end
    end
end

### MERGE PROPOSAL ###
@gen function merge_proposal(tr, is_smart)
    current_num_rels = tr[:num_rels]
    to_idx ~ uniform_discrete(1, current_num_rels - 1)
    from_idx1 ~ uniform_discrete(1, current_num_rels)
    if is_smart
        from_idx2 = {*} ~ smart_sample_second_relation_to_merge(tr, from_idx1, current_num_rels)
    else
        from_idx2 ~ uniform_from_list([idx for idx=1:current_num_rels if idx != from_idx1])
    end
end

@gen function smart_sample_second_relation_to_merge(tr, from_idx1, current_num_rels)
    function cnt(rel)
        if haskey(tr[:verbs => :counts], rel)
            tr[:verbs => :counts][rel]
        else
            zeros(num_verbs(tr))
        end
    end
    first_rel = from_idx1
    other_indices = [x for x=1:current_num_rels if x !== from_idx1]
    if length(other_indices) == 0
        println("0 other possibilities")
        println("Currently $(tr[:world => :num_relations => ()]) relations in the trace.")
        println("First rel is $from_idx1")
    end
    factors = [
        logbeta(cnt(other_rel) + cnt(first_rel) .+ dirichlet_prior_val(tr)) for other_rel in other_indices
    ]
    logprobs = factors .- logsumexp(factors)
    probs = exp.(logprobs)
    if !isapprox(sum(probs), 1.)
        println("Factors: $factors")
        println("Prob: $probs")
    end
    val = {:from_idx2} ~ categorical_from_list(other_indices, probs)
end

### INVOLUTION ###
@involution function splitmerge_involution()
    do_split = @read_discrete_from_proposal(:do_split)
    @write_discrete_to_proposal(:do_split, !do_split)

    if do_split
        @invcall split_transformation()
    else
        @invcall merge_transformation()
    end
end

function get_split_idx_transformation(nrels, from_idx, new_idx1, new_idx2)
    new_to_old = convert(Vector{Union{Nothing, Int}}, [i for i=1:nrels])
    deleteat!(new_to_old, from_idx)
    smaller = min(new_idx1, new_idx2)
    bigger = max(new_idx1, new_idx2)
    insert!(new_to_old, smaller, nothing)
    insert!(new_to_old, bigger, nothing)
    return new_to_old
end

function get_merge_idx_transformation(nrels, from_idx1, from_idx2, new_idx)
    new_to_old = convert(Vector{Union{Nothing, Int}}, [i for i=1:nrels])
    deleteat!(new_to_old, max(from_idx1, from_idx2))
    deleteat!(new_to_old, min(from_idx1, from_idx2))
    insert!(new_to_old, new_idx, nothing)
    return new_to_old
end

@involution function split_transformation()
    from_idx = @read_discrete_from_proposal(:from_idx)
    new_idx1 = @read_discrete_from_proposal(:new_idx1)
    new_idx2 = @read_discrete_from_proposal(:new_idx2)
    @write_discrete_to_proposal(:from_idx1, new_idx1)
    @write_discrete_to_proposal(:from_idx2, new_idx2)
    @write_discrete_to_proposal(:to_idx, from_idx)

    ### DO SPLIT ###
    # must move:
    # fact truth values
    # samples for sentences
    current_n_rels = @read_discrete_from_model(:num_rels)
    new_idx_to_old_idx = get_split_idx_transformation(current_n_rels, from_idx, new_idx1, new_idx2)
    for (newidx, oldidx) in enumerate(new_idx_to_old_idx)
        oldidx === nothing && continue
        # move fact values
        entpair_subset = @read_discrete_from_model(:sampled_facts => :all_facts => :facts_per_rel => oldidx => :true_entpairs)
        @write_discrete_to_model(:sampled_facts => :all_facts => :facts_per_rel => newidx => :true_entpairs, entpair_subset)

        # move samples for sentences
        entity_to_indices = @read_discrete_from_model(:verbs => :indices_per_entity)
        inds = haskey(entity_to_indices, oldidx) ? entity_to_indices[oldidx] : ()
        for idx in inds
            fact = @read_discrete_from_model(:sampled_facts => :sampled_facts => idx)
            @assert fact.rel == oldidx
            @write_discrete_to_model(:sampled_facts => :sampled_facts => idx, Fact(newidx, fact.ent1, fact.ent2))
        end
    end

    @write_discrete_to_model(:num_rels, current_n_rels + 1)
    @invcall handle_split_reassocs(from_idx, new_idx1, new_idx2)
end

@involution function handle_split_reassocs(from_idx, new_idx1, new_idx2)
    true1 = Set()
    true2 = Set()
    indices_per_entity = @read_discrete_from_model(:verbs => :indices_per_entity)
    old_rel_inds = haskey(indices_per_entity, from_idx) ? indices_per_entity[from_idx] : ()
    for idx in old_rel_inds
        fact = @read_discrete_from_model(:sampled_facts => :sampled_facts => idx)
        entpair = (fact.ent1, fact.ent2)
        to_first = @read_discrete_from_proposal(:to_first => idx)
        if to_first
            push!(true1, entpair)
            @write_discrete_to_model(:sampled_facts => :sampled_facts => idx, Fact(new_idx1, entpair...))
        else
            push!(true2, entpair)
            @write_discrete_to_model(:sampled_facts => :sampled_facts => idx, Fact(new_idx2, entpair...))
        end
    end

    # TODO: if I had save for reverse regenerate, I could more closely match the full inference program!
    @write_discrete_to_model(:sampled_facts => :all_facts => :facts_per_rel => new_idx1 => :true_entpairs, true1)
    @write_discrete_to_model(:sampled_facts => :all_facts => :facts_per_rel => new_idx2 => :true_entpairs, true2)
end

@involution function merge_transformation()
    to_idx = @read_discrete_from_proposal(:to_idx)
    from_idx1 = @read_discrete_from_proposal(:from_idx1)
    from_idx2 = @read_discrete_from_proposal(:from_idx2)
    @write_discrete_to_proposal(:new_idx1, from_idx1)
    @write_discrete_to_proposal(:new_idx2, from_idx2)
    @write_discrete_to_proposal(:from_idx, to_idx)

    ### DO MERGE ###
    # must move:
    # fact truth values
    # samples for sentences
    current_n_rels = @read_discrete_from_model(:num_rels)
    new_idx_to_old_idx = get_merge_idx_transformation(current_n_rels, from_idx1, from_idx2, to_idx)
    for (newidx, oldidx) in enumerate(new_idx_to_old_idx)
        oldidx === nothing && continue
        # move fact values
        entpair_subset = @read_discrete_from_model(:sampled_facts => :all_facts => :facts_per_rel => oldidx => :true_entpairs)
        @write_discrete_to_model(:sampled_facts => :all_facts => :facts_per_rel => newidx => :true_entpairs, entpair_subset)

        # move samples for sentences
        entity_to_indices = @read_discrete_from_model(:verbs => :indices_per_entity)
        inds = haskey(entity_to_indices, oldidx) ? entity_to_indices[oldidx] : ()
        for idx in inds
            fact = @read_discrete_from_model(:sampled_facts => :sampled_facts => idx)
            @assert fact.rel == oldidx
            @write_discrete_to_model(:sampled_facts => :sampled_facts => idx, Fact(newidx, fact.ent1, fact.ent2))
        end
    end

    @write_discrete_to_model(:num_rels, current_n_rels - 1)
    @invcall handle_merge_reassocs(from_idx1, from_idx2, to_idx)
end

@involution function handle_merge_reassocs(from_idx1, from_idx2, to_idx)
    indices_per_entity = @read_discrete_from_model(:verbs => :indices_per_entity)
    inds1 = haskey(indices_per_entity, from_idx1) ? indices_per_entity[from_idx1] : ()
    inds2 = haskey(indices_per_entity, from_idx2) ? indices_per_entity[from_idx2] : ()
    inds = Iterators.flatten((inds1, inds2))
    trues = Set();
    for idx in inds
        fact = @read_discrete_from_model(:sampled_facts => :sampled_facts => idx)
        @assert fact.rel == from_idx1 || fact.rel == from_idx2
        entpair = (fact.ent1, fact.ent2)
        push!(trues, entpair)
        @write_discrete_to_model(:sampled_facts => :sampled_facts => idx, Fact(to_idx, entpair...))

        # println("writing bwd for  :to_first => $idx")
        if idx in inds1
            @write_discrete_to_proposal(:to_first => idx, true)
        else
            @write_discrete_to_proposal(:to_first => idx, false)
        end
    end
    @write_discrete_to_model(:sampled_facts => :all_facts => :facts_per_rel => to_idx => :true_entpairs, trues)
end

#######################
# Put it all together #
#######################
@gen function splitmerge_proposal(tr, smartness_prior)
    do_smart ~ bernoulli(smartness_prior)

    current_num_rels = tr[:num_rels]
    priorval = current_num_rels == 1 ? 1. : 0.5
    do_split ~ bernoulli(priorval)

    if do_split
        {*} ~ split_proposal(tr, do_smart)
    else
        {*} ~ merge_proposal(tr, do_smart)
    end
end
@gen function sdds_splitmerge_proposal(tr)
    {*} ~ splitmerge_proposal(tr, 0.5)
end

@involution function sdds_splitmerge_involution(_, _, _)
    do_smart = @read_discrete_from_proposal(:do_smart)
    @write_discrete_to_proposal(:do_smart, !do_smart)
    @invcall splitmerge_involution()
end