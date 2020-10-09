### SPLIT PROPOSAL ###
@gen function split_proposal(tr, is_smart)
    current_num_rels = tr[:world => :num_relations => ()]
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
        if haskey(tr[:kernel => :verbs => :counts], rel)
            tr[:kernel => :verbs => :counts][rel]
        else
            zeros(num_verbs(tr))
        end
    end
    logfactors = [
        -logbeta(cnt(rel) .+ dirichlet_prior_val(tr))
        for rel in map(idx -> GenWorldModels.convert_to_abstract(tr.world, Relation(idx)), 1:current_num_rels)
    ]
    logprobs = logfactors .- logsumexp(logfactors)
    probs = exp.(logprobs)
    return {:from_idx} ~ categorical(probs)
end

@gen function sample_split_assignments(tr, from_idx, is_smart)
    abstract_rel = GenWorldModels.convert_to_abstract(tr.world, Relation(from_idx))
    count1 = zeros(Int, num_verbs(tr))
    count2 = zeros(Int, num_verbs(tr))
    itr = haskey(tr[:kernel => :verbs => :indices_per_entity], abstract_rel) ? tr[:kernel => :verbs => :indices_per_entity][abstract_rel] : ()
    for idx in itr
        mention = verbs(tr)[idx]
        if is_smart
            p1 = (dirichlet_prior_val(tr) + count1[mention]) / (num_verbs(tr) * dirichlet_prior_val(tr) + sum(count1))
            p2 = (dirichlet_prior_val(tr) + count2[mention]) / (num_verbs(tr) * dirichlet_prior_val(tr) + sum(count2))
            to_first = {:to_first => idx} ~ bernoulli(p1/(p1+p2))
        else
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
    current_num_rels = tr[:world => :num_relations => ()]
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
        if haskey(tr[:kernel => :verbs => :counts], rel)
            tr[:kernel => :verbs => :counts][rel]
        else
            zeros(num_verbs(tr))
        end
    end
    first_rel = GenWorldModels.convert_to_abstract(tr.world, Relation(from_idx1))
    other_indices = [x for x=1:current_num_rels if x !== from_idx1]
    if length(other_indices) == 0
        println("0 other possibilities")
        println("Currently $(tr[:world => :num_relations => ()]) relations in the trace.")
        println("First rel is $from_idx1")
    end
    abstract_other_rels = [GenWorldModels.convert_to_abstract(tr.world, Relation(x)) for x in other_indices]
    factors = [
        logbeta(cnt(other_rel) + cnt(first_rel) .+ dirichlet_prior_val(tr)) for other_rel in abstract_other_rels
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
@oupm_involution splitmerge_involution (old, fwd) to (new, bwd) begin
    do_split = @read(fwd[:do_split], :disc)
    @write(bwd[:do_split], !do_split, :disc)

    if do_split
        @tcall split_transformation()
    else
        @tcall merge_transformation()
    end
end

@oupm_involution split_transformation (old, fwd) to (new, bwd) begin
    from_idx = @read(fwd[:from_idx], :disc)
    new_idx1 = @read(fwd[:new_idx1], :disc)
    new_idx2 = @read(fwd[:new_idx2], :disc)
    @write(bwd[:from_idx1], new_idx1, :disc)
    @write(bwd[:from_idx2], new_idx2, :disc)
    @write(bwd[:to_idx], from_idx, :disc)

    @split(Relation(from_idx), new_idx1, new_idx2)
    num_r = @read(old[:world => :num_relations => () => :num], :disc) + 1
    @write(new[:world => :num_relations => () => :num], num_r, :disc)

    @tcall handle_split_reassocs(from_idx, new_idx1, new_idx2)
end
@oupm_involution do_split(from_idx, new_idx) (old, fwd) to (new, bwd) begin
end
@oupm_involution handle_split_reassocs(from_idx, new_idx1, new_idx2) (old, fwd) to (new, bwd) begin
    old_rel = @convert_to_abstract(Relation(from_idx))
    true1 = Set()
    true2 = Set()
    indices_per_entity = @read(old[:kernel => :verbs => :indices_per_entity], :disc)
    old_rel_inds = haskey(indices_per_entity, old_rel) ? indices_per_entity[old_rel] : ()
    for idx in old_rel_inds
        fact = @read(old[:kernel => :sampled_facts => :sampled_facts => idx], :disc)
        entpair = (fact.ent1, fact.ent2)
        to_first = @read(fwd[:to_first => idx], :disc)
        if to_first
            push!(true1, entpair)
            @write(new[:kernel => :sampled_facts => :sampled_facts => idx], Fact(Relation(new_idx1), entpair...), :disc)
        else
            push!(true2, entpair)
            @write(new[:kernel => :sampled_facts => :sampled_facts => idx], Fact(Relation(new_idx2), entpair...), :disc)
        end
    end
    for ep in true1
        @write(new[:kernel => :sampled_facts => :all_facts => :rels_to_facts => Relation(new_idx1) => :true_entpairs => ep], true, :disc)
    end
    for ep in true2
        @write(new[:kernel => :sampled_facts => :all_facts => :rels_to_facts => Relation(new_idx2) => :true_entpairs => ep], true, :disc)
    end
    bwd_unconstrained = invert(select(Iterators.flatten((true1, true2))...))
    @save_for_reverse_regenerate(:kernel => :sampled_facts => :all_facts => :rels_to_facts => Relation(from_idx) => :true_entpairs, bwd_unconstrained)
end

@oupm_involution merge_transformation (old, fwd) to (new, bwd) begin
    idx1 = @read(fwd[:from_idx1], :disc)
    idx2 = @read(fwd[:from_idx2], :disc)
    to_idx = @read(fwd[:to_idx], :disc)
    @write(bwd[:new_idx1], idx1, :disc)
    @write(bwd[:new_idx2], idx2, :disc)
    @write(bwd[:from_idx], to_idx, :disc)

    @merge(Relation(to_idx), idx1, idx2)
    num_r = @read(old[:world => :num_relations => () => :num], :disc) - 1
    @write(new[:world => :num_relations => () => :num], num_r, :disc)

    @tcall handle_merge_reassocs(idx1, idx2, to_idx)
end
@oupm_involution handle_merge_reassocs(idx1, idx2, to_idx) (old, fwd) to (new, bwd) begin
    rel1 = @convert_to_abstract(Relation(idx1))
    rel2 = @convert_to_abstract(Relation(idx2))
    ent_to_idx =  @read(old[:kernel => :verbs => :indices_per_entity], :disc)
    inds1 = haskey(ent_to_idx, rel1) ? ent_to_idx[rel1] : ()
    inds2 = haskey(ent_to_idx, rel2) ? ent_to_idx[rel2] : ()
    sentence_indices = Iterators.flatten((inds1, inds2))
    trues = Set()
    true1 = Set()
    true2 = Set()
    for idx in sentence_indices
        fact = @read(old[:kernel => :sampled_facts => :sampled_facts => idx], :disc)
        entpair = (fact.ent1, fact.ent2)
        push!(trues, entpair)
        @write(new[:kernel => :sampled_facts => :sampled_facts => idx], Fact(Relation(to_idx), entpair...), :disc)

        if idx in inds1
            @write(bwd[:to_first => idx], true, :disc)
            push!(true1, entpair)
        end
        if idx in inds2
            @write(bwd[:to_first => idx], false, :disc)
            push!(true2, entpair)
        end
    end

    for entpair in trues
        @write(new[:kernel => :sampled_facts => :all_facts => :rels_to_facts => Relation(to_idx) => :true_entpairs => entpair], true, :disc)
    end

    bwd_unconstrained1 = invert(select(true1...))
    bwd_unconstrained2 = invert(select(true2...))
    @save_for_reverse_regenerate(:kernel => :sampled_facts => :all_facts => :rels_to_facts => Relation(idx1) => :true_entpairs, bwd_unconstrained1)
    @save_for_reverse_regenerate(:kernel => :sampled_facts => :all_facts => :rels_to_facts => Relation(idx2) => :true_entpairs, bwd_unconstrained2)
end

#######################
# Put it all together #
#######################
@gen function splitmerge_proposal(tr, smartness_prior)
    do_smart ~ bernoulli(smartness_prior)

    current_num_rels = tr[:world => :num_relations => ()]
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

@oupm_involution sdds_splitmerge_involution (old, fwd) to (new, bwd) begin
    do_smart = @read(fwd[:do_smart], :disc)
    @write(bwd[:do_smart], !do_smart, :disc)
    @tcall splitmerge_involution()
end

sdds_splitmerge_kernel = OUPMMHKernel(sdds_splitmerge_proposal, (), sdds_splitmerge_involution)