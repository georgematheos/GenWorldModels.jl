##################
# SPLIT PROPOSAL #
##################

@gen function split_proposal(tr, is_smart)
    num_entities = get_args(tr)[1]
    num_verbs = get_args(tr)[2]
    current_num_rels = tr[:world => :num_relations => ()]
    to_idx1 ~ uniform_discrete(1, current_num_rels + 1)
    to_idx2 ~ uniform_from_list([idx for idx=1:(current_num_rels + 1) if idx != to_idx1])

    if is_smart
        from_idx = {*} ~ sample_smartsplit_relation(tr, current_num_rels)
    else
        from_idx ~ uniform_discrete(1, current_num_rels)
    end

    (count1, count2, true1pairs, true2pairs) = {*} ~ sample_split_assignments(tr, from_idx, is_smart)

    (true1count, true2count) = {*} ~ sample_fact_truth_values(tr, from_idx, to_idx1, to_idx2, true1pairs, true2pairs, num_entities)
    {*} ~ sample_split_sparsities(to_idx1, to_idx2, true1count, true2count, num_entities)
    {*} ~ sample_split_dirichlets(to_idx1, to_idx2, count1, count2)
end

@gen function sample_smartsplit_relation(tr, current_num_rels)
    num_verbs = get_args(tr)[2]
    num_m_per_e = tr[:kernel => :counts].num_mentions_per_entity
    cnt(rel) = haskey(num_m_per_e, rel) ? num_m_per_e[rel] : zeros(num_verbs)
    factors = [
        exp(-logbeta(cnt(rel) .+ DIRICHLET_PRIOR_VAL))
        for rel in map(idx -> GenWorldModels.convert_to_abstract(tr.world, Relation(idx)), 1:current_num_rels)
    ]
    probs = factors/sum(factors)
    from_idx ~ categorical(probs)
    # println("Going to split $from_idx which currently has mentions ", cnt(GenWorldModels.convert_to_abstract(tr.world, Relation(from_idx))))
    return from_idx
end

@gen function sample_split_assignments(tr, from_idx, is_smart)
    num_verbs = get_args(tr)[2]
    abst = GenWorldModels.convert_to_abstract(tr.world, Relation(from_idx))
    count1 = zeros(Int, num_verbs)
    count2 = zeros(Int, num_verbs)
    true1pairs = Set()
    true2pairs = Set()
    inds_per_ent = tr[:kernel => :counts].indices_per_entity
    if haskey(inds_per_ent, abst)
        for idx in inds_per_ent[abst]
            sentence = get_retval(tr)[idx]
            (ent1, mention, ent2) = sentence
            if is_smart
                p1 = (DIRICHLET_PRIOR_VAL + count1[mention]) / (num_verbs*DIRICHLET_PRIOR_VAL + sum(count1))
                p2 = (DIRICHLET_PRIOR_VAL + count2[mention]) / (num_verbs*DIRICHLET_PRIOR_VAL + sum(count2))
                to_lower = {:to_lower => idx} ~ bernoulli(p1/(p1 + p2))
            else
                to_lower = {:to_lower => idx} ~ bernoulli(0.5)
            end
            if to_lower
                push!(true1pairs, (ent1, ent2))
                count1[mention] += 1
            else
                push!(true2pairs, (ent1, ent2))
                count2[mention] += 1
            end
        end
    end

    # println("After the split, the mentions will be $count1 and $count2")

    return (count1, count2, true1pairs, true2pairs)
end

@gen function sample_fact_truth_values(tr, from_idx, to_idx1, to_idx2, true1pairs, true2pairs, num_entities)
    sparsity = tr[:world => :sparsity => Relation(from_idx)]
    num1true = 0
    num2true = 0
    for e1=1:num_entities, e2=1:num_entities
        sp1 = (e1, e2) in true1pairs ? 1. : sparsity
        sp2 = (e1, e2) in true2pairs ? 1. : sparsity
        istrue1 = {:world => :num_facts => (Relation(to_idx1), Entity(e1), Entity(e2)) => :is_true} ~ bernoulli(sp1)
        istrue2 = {:world => :num_facts => (Relation(to_idx2), Entity(e1), Entity(e2)) => :is_true} ~ bernoulli(sp2)
        if istrue1; num1true += 1; end
        if istrue2; num2true += 1; end
    end

    return (num1true, num2true)
end

@gen function sample_split_sparsities(to_idx1, to_idx2, true1count, true2count, num_entities)
    {:world => :sparsity => Relation(to_idx1) => :sparsity} ~ beta(BETA_PRIOR[1] + true1count, BETA_PRIOR[2] + num_entities^2 - true1count)
    {:world => :sparsity => Relation(to_idx2) => :sparsity} ~ beta(BETA_PRIOR[1] + true2count, BETA_PRIOR[2] + num_entities^2 - true2count)
end

@gen function sample_split_dirichlets(to_idx1, to_idx2, count1, count2)
    if sum(count1) > 0
        {:world => :verb_prior => Relation(to_idx1) => :prior} ~ dirichlet(count1 .+ DIRICHLET_PRIOR_VAL)
    end
    if sum(count2) > 0
        {:world => :verb_prior => Relation(to_idx2) => :prior} ~ dirichlet(count2 .+ DIRICHLET_PRIOR_VAL)
    end
end

##################
# MERGE PROPOSAL #
##################

@gen function merge_proposal(tr, is_smart)
    current_num_rels = tr[:world => :num_relations => ()]
    from_idx1 ~ uniform_discrete(1, current_num_rels)
    to_idx ~ uniform_discrete(1, current_num_rels - 1)
    if is_smart
        from_idx2 = {*} ~ smart_sample_second_relation_to_merge(tr, from_idx1, current_num_rels)
    else
        from_idx2 ~ uniform_from_list([idx for idx=1:current_num_rels if idx != from_idx1])
    end

    (guaranteed_true_entpairs, merged_count) = find_references_during_merge(tr, from_idx1, from_idx2)
    num_true_entpairs = {*} ~ sample_merge_fact_vals(tr, from_idx1, from_idx2, to_idx, guaranteed_true_entpairs)
    {*} ~ sample_merge_sparsity(to_idx, num_true_entpairs, get_args(tr)[1])
    {*} ~ sample_merge_verbprior(to_idx, merged_count)
end

@gen function smart_sample_second_relation_to_merge(tr, from_idx1, current_num_rels)
    num_verbs = get_args(tr)[2]
    α = fill(DIRICHLET_PRIOR_VAL, num_verbs)
    abst = GenWorldModels.convert_to_abstract(tr.world, Relation(from_idx1))
    counts = (tr[:kernel => :counts]::EntityMentionCounts).num_mentions_per_entity
    other_rel_indices = [r for r=1:current_num_rels if r != from_idx1]
    abstract_other_rels = map(x -> GenWorldModels.convert_to_abstract(tr.world, Relation(x)), other_rel_indices)
    
    cnt(rel) = haskey(counts, rel) ? counts[rel] : zeros(num_verbs)

    factors = [
        exp(logbeta(α + cnt(other_rel) + cnt(abst))) for other_rel in abstract_other_rels
    ]
    probs = factors/sum(factors)

    from_idx2 ~ categorical_from_list(other_rel_indices, probs)
    return from_idx2
end

function find_references_during_merge(tr, from_idx1, from_idx2)
    num_verbs = get_args(tr)[2]
    abst1 = GenWorldModels.convert_to_abstract(tr.world, Relation(from_idx1))
    abst2 = GenWorldModels.convert_to_abstract(tr.world, Relation(from_idx2))
    guaranteed_true_entpairs = Set()
    merged_count = zeros(Int, num_verbs)
    inds_per_e = tr[:kernel => :counts].indices_per_entity
    if haskey(inds_per_e, abst1)
        for idx in inds_per_e[abst1]
            (ent1, mention, ent2) = get_retval(tr)[idx]
            push!(guaranteed_true_entpairs, (ent1, ent2))
            merged_count[mention] += 1
        end
    end
    if haskey(inds_per_e, abst2)
        for idx in inds_per_e[abst2]
            (ent1, mention, ent2) = get_retval(tr)[idx]
            push!(guaranteed_true_entpairs, (ent1, ent2))
            merged_count[mention] += 1
        end
    end

    return (guaranteed_true_entpairs, merged_count)
end

@gen function sample_merge_fact_vals(tr, from_idx1, from_idx2, to_idx, guaranteed_true_entpairs)
    num_entities = get_args(tr)[1]
    sp1 = tr[:world => :sparsity => Relation(from_idx1)]
    sp2 = tr[:world => :sparsity => Relation(from_idx2)]
    sp = (sp1 + sp2)/2
    num_true = 0
    for e1=1:num_entities, e2=1:num_entities
        p = (e1, e2) in guaranteed_true_entpairs ? 1. : sp
        is_true = {:world => :num_facts => (Relation(to_idx), Entity(e1), Entity(e2)) => :is_true} ~ bernoulli(p)
        if is_true
            num_true += 1
        end
    end
    return num_true
end

@gen function sample_merge_sparsity(to_idx, num_true, num_entities)
    {:world => :sparsity => Relation(to_idx) => :sparsity} ~ beta(BETA_PRIOR[1] + num_true, BETA_PRIOR[2] + num_entities^2 - num_true)
end

@gen function sample_merge_verbprior(to_idx, count)
    if sum(count) > 0
        {:world => :verb_prior => Relation(to_idx) => :prior} ~ dirichlet(count .+ DIRICHLET_PRIOR_VAL)
    end
end

#########################
# SPLITMERGE INVOLUTION #
#########################

@oupm_involution splitmerge_involution(num_entities) (old, fwd) to (new, bwd) begin
    do_split = @read(fwd[:do_split], :disc)
    @write(bwd[:do_split], !do_split, :disc)

    if do_split
        @tcall split_transformation(num_entities)
    else
        @tcall merge_transformation(num_entities)
    end
end

@oupm_involution split_transformation(num_entities) (old, fwd) to (new, bwd) begin
    (from_idx, to1, to2) = @tcall handle_split_indices()
    @tcall do_split(from_idx, to1, to2, num_entities)
    @tcall handle_assignments_verbpriors_truthvals_during_split(from_idx, to1, to2, num_entities)
end
@oupm_involution handle_split_indices (old, fwd) to (new, bwd) begin
    from_idx = @read(fwd[:from_idx], :disc)
    to1 = @read(fwd[:to_idx1], :disc)
    to2 = @read(fwd[:to_idx2], :disc)
    @write(bwd[:from_idx1], to1, :disc)
    @write(bwd[:from_idx2], to2, :disc)
    @write(bwd[:to_idx], from_idx, :disc)

    return (from_idx, to1, to2)
end
@oupm_involution do_split(from_idx, to1, to2, num_entities) (old, fwd) to (new, bwd) begin
    moves = Tuple(
        Fact((Relation(from_idx), Entity(e1), Entity(e2)), 1) => nothing for e1=1:num_entities, e2=1:num_entities
        if @read(old[:world => :num_facts => (Relation(from_idx), Entity(e1), Entity(e2)) => :is_true], :disc)
    )
    @split(Relation(from_idx), to1, to2, moves)
    num_r = @read(old[:world => :num_relations => () => :num], :disc) + 1
    @write(new[:world => :num_relations => () => :num], num_r, :disc)
end

@oupm_involution handle_assignments_verbpriors_truthvals_during_split(from_idx, to1, to2, num_entities) (old, fwd) to (new, bwd) begin
    abst_rel = @convert_to_abstract(Relation(from_idx))
    if haskey(@read(old[:kernel => :counts], :disc).indices_per_entity, abst_rel)
        for idx in @read(old[:kernel => :counts], :disc).indices_per_entity[abst_rel]
            new_idx = @read(fwd[:to_lower => idx], :disc) ? to1 : to2
            (_, (ent1, mention, ent2)) = @read(old[:kernel => :rels_and_sentences => idx], :disc)
            @write(
                new[:kernel => :facts => :sampled_facts => idx],
                Fact((Relation(new_idx), Entity(ent1), Entity(ent2)), 1),
                :disc
            )
        end
        @copy(old[:world => :verb_prior => Relation(from_idx)], bwd[:world => :verb_prior => Relation(from_idx)])
    end
    @copy(fwd[:world => :sparsity], new[:world => :sparsity])
    @copy(fwd[:world => :num_facts], new[:world => :num_facts])
    @copy(fwd[:world => :verb_prior], new[:world => :verb_prior])
    @copy(old[:world => :sparsity => Relation(from_idx)], bwd[:world => :sparsity => Relation(from_idx)])

    for e1=1:num_entities, e2=1:num_entities
        a = :world => :num_facts => (Relation(from_idx), Entity(e1), Entity(e2))
        @copy(old[a], bwd[a])
    end
end

@oupm_involution merge_transformation(num_entities) (old, fwd) to (new, bwd) begin
    (to_idx, from1, from2) = @tcall handle_merge_indices()
    @tcall do_merge(to_idx, from1, from2, num_entities)
    @tcall handle_assignments_verbpriors_truthvals_sparsities_during_merge(to_idx, from1, from2, num_entities)
end

@oupm_involution handle_merge_indices (old, fwd) to (new, bwd) begin
    to_idx = @read(fwd[:to_idx], :disc)
    from1 = @read(fwd[:from_idx1], :disc)
    from2 = @read(fwd[:from_idx2], :disc)
    @write(bwd[:to_idx1], from1, :disc)
    @write(bwd[:to_idx2], from2, :disc)
    @write(bwd[:from_idx], to_idx, :disc)

    return (to_idx, from1, from2)
end
@oupm_involution do_merge(to_idx, from1, from2, num_entities) (old, fwd) to (new, bwd) begin
    moves = Tuple(
        Iterators.flatten(
            (
                (
                    Fact((Relation(from1), Entity(e1), Entity(e2)), 1) => nothing
                    for e1=1:num_entities, e2=1:num_entities
                    if @read(old[:world => :num_facts => (Relation(from1), Entity(e1), Entity(e2)) => :is_true], :disc)
                ),
                (
                    Fact((Relation(from2), Entity(e1), Entity(e2)), 1) => nothing
                    for e1=1:num_entities, e2=1:num_entities
                    if @read(old[:world => :num_facts => (Relation(from2), Entity(e1), Entity(e2)) => :is_true], :disc)
                )
            )
        )
    )
    @merge(Relation(to_idx), from1, from2, moves)
    @write(new[:world => :num_relations => () => :num], @read(old[:world => :num_relations => () => :num], :disc) - 1, :disc)
end
@oupm_involution handle_assignments_verbpriors_truthvals_sparsities_during_merge(to_idx, from1, from2, num_entities) (old, fwd) to (new, bwd) begin
    abst_rel1 = @convert_to_abstract(Relation(from1))
    abst_rel2 = @convert_to_abstract(Relation(from2))
    ipe = @read(old[:kernel => :counts], :disc).indices_per_entity
    if haskey(ipe, abst_rel1)
        for idx in ipe[abst_rel1]
            (_, (ent1, mention, ent2)) = @read(old[:kernel => :rels_and_sentences => idx], :disc)

            @write(
                new[:kernel => :facts => :sampled_facts => idx],
                Fact((Relation(to_idx), Entity(ent1), Entity(ent2)), 1),
                :disc
            )
            @write(bwd[:to_lower => idx], true, :disc)
        end
        @copy(old[:world => :verb_prior => Relation(from1)], bwd[:world => :verb_prior => Relation(from1)])
    end
    if haskey(ipe, abst_rel2)
        for idx in ipe[abst_rel2]
            (_, (ent1, mention, ent2)) = @read(old[:kernel => :rels_and_sentences => idx], :disc)
            @write(
                new[:kernel => :facts => :sampled_facts => idx],
                Fact((Relation(to_idx), Entity(ent1), Entity(ent2)), 1),
                :disc
            )
            @write(bwd[:to_lower => idx], false, :disc)
        end
        @copy(old[:world => :verb_prior => Relation(from2)], bwd[:world => :verb_prior => Relation(from2)])
    end
    @copy(fwd[:world => :sparsity], new[:world => :sparsity])
    @copy(fwd[:world => :num_facts], new[:world => :num_facts])
    @copy(fwd[:world => :verb_prior], new[:world => :verb_prior])

    @copy(old[:world => :sparsity => Relation(from1)], bwd[:world => :sparsity => Relation(from1)])
    @copy(old[:world => :sparsity => Relation(from2)], bwd[:world => :sparsity => Relation(from2)])

    for e1=1:num_entities, e2=1:num_entities
        a1 = :world => :num_facts => (Relation(from1), Entity(e1), Entity(e2))
        a2 = :world => :num_facts => (Relation(from2), Entity(e1), Entity(e2))
        @copy(old[a1], bwd[a1])
        @copy(old[a2], bwd[a2])
    end
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
@gen function dumb_splitmerge_proposal(tr)
    {*} ~ splitmerge_proposal(tr, 0.)
end
@gen function smart_splitmerge_proposal(tr)
    {*} ~ splitmerge_proposal(tr, 1.)
end

@oupm_involution sdds_splitmerge_involution (old, fwd) to (new, bwd) begin
    do_smart = @read(fwd[:do_smart], :disc)
    @write(bwd[:do_smart], !do_smart, :disc)
    num_entities = @read(old[:world => :args => :num_entities], :disc)
    @tcall splitmerge_involution(num_entities)
end

@oupm_involution samesmartness_splitmerge_involution (old, fwd) to (new, bwd) begin
    do_smart = @read(fwd[:do_smart], :disc)
    @write(bwd[:do_smart], do_smart, :disc)
    num_entities = @read(old[:world => :args => :num_entities], :disc)
    @tcall splitmerge_involution(num_entities)
end

sdds_splitmerge_kernel = OUPMMHKernel(sdds_splitmerge_proposal, (), sdds_splitmerge_involution)
dumb_splitmerge_kernel = OUPMMHKernel(dumb_splitmerge_proposal, (), samesmartness_splitmerge_involution)
smart_splitmerge_kernel = OUPMMHKernel(smart_splitmerge_proposal, (), samesmartness_splitmerge_involution)