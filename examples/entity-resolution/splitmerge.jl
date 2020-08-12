@gen function dumb_split_proposal(tr)
    num_entities = get_args(tr)[1]
    current_num_rels = tr[:world => :num_relations => ()]
    from_idx ~ uniform_discrete(1, current_num_rels)
    to_idx1 ~ uniform_discrete(1, current_num_rels)
    to_idx2 ~ uniform_discrete(to_idx1 + 1, current_num_rels + 1)

    abst_rel = GenWorldModels.convert_to_abstract(tr.world, Relation(from_idx))
    truthiness = Dict()
    num_true_1 = 0
    num_true_2 = 0
    for e1=1:num_entities, e2=1:num_entities
        if tr[:world => :num_facts => (Relation(from_idx), Entity(e1), Entity(e2))]
            # 1 = only to1 true; 2 = only to2 true, 3 = both true
            t = {:truthiness => (e1, e2)} ~ uniform_discrete(1, 3)
            truthiness[(e1, e2)] = t
            if t == 1 || t == 3
                num_true_1 += 1
            end
            if t == 2 || t == 3
                num_true_2 += 1
            end
        end
    end

    num_verbs = get_args(tr)[2]
    count1 = zeros(Int, num_verbs)
    count2 = zeros(Int, num_verbs)
    if haskey(tr[:kernel => :counts].indices_per_entity, abst_rel)
        for idx in tr[:kernel => :counts].indices_per_entity[abst_rel]
            sentence = get_retval(tr)[idx]
            e1, e2 = sentence[1], sentence[3]
            if truthiness[(e1, e2)] == 1
                prior = 1.
            elseif truthiness[(e1, e2)] == 2
                prior = 0.
            else
                prior = 0.5
            end
            to_lower = {:to_lower => idx} ~ bernoulli(prior)
            mention = sentence[2]
            if to_lower
                count1[mention] += 1
            else
                count2[mention] += 1
            end
        end
    end
    if sum(count1) > 0
        {:verb_prior => Relation(to_idx1) => :prior} ~ dirichlet(count1 .+ DIRICHLET_PRIOR_VAL)
    end
    if sum(count2) > 0
        {:verb_prior => Relation(to_idx2) => :prior} ~ dirichlet(count1 .+ DIRICHLET_PRIOR_VAL)
    end


    {:sparsity => Relation(to_idx1) => :sparsity} ~ beta(BETA_PRIOR[1] + num_true_1, BETA_PRIOR[2] + num_entities^2)
    {:sparsity => Relation(to_idx2) => :sparsity} ~ beta(BETA_PRIOR[1] + num_true_2, BETA_PRIOR[2] + num_entities^2)
end

@gen function dumb_merge_proposal(tr)
    current_num_rels = tr[:world => :num_relations => ()]
    from_idx1 ~ uniform_discrete(1, current_num_rels - 1)
    from_idx2 ~ uniform_discrete(from_idx1 + 1, current_num_rels)
    to_idx ~ uniform_discrete(1, current_num_rels - 1)

    num_verbs = get_args(tr)[2]
    count = zeros(Int, num_verbs)
    abst1 = GenWorldModels.convert_to_abstract(tr.world, Relation(from_idx1))
    abst2 = GenWorldModels.convert_to_abstract(tr.world, Relation(from_idx2))
    if haskey(tr[:kernel => :counts].indices_per_entity, abst1)
        for idx in tr[:kernel => :counts].indices_per_entity[abst1]
            mention = get_retval(tr)[idx][2]
            count[mention] += 1
        end
    end
    if haskey(tr[:kernel => :counts].indices_per_entity, abst2)
        for idx in tr[:kernel => :counts].indices_per_entity[abst2]
            mention = get_retval(tr)[idx][2]
            count[mention] += 1
        end
    end
    if sum(count) > 0
        {:verb_prior => Relation(to_idx) => :prior} ~ dirichlet(count .+ DIRICHLET_PRIOR_VAL)
    end
    num_entities = get_args(tr)[1]
    facts1 = tr[:kernel => :facts => :facts => :facts_per_rel => from_idx1 => :facts]
    facts2 = tr[:kernel => :facts => :facts => :facts_per_rel => from_idx2 => :facts]
    num_true = length([1 for idx=1:num_entities^2 if !isempty(facts1[idx]) || !isempty(facts2[idx])])
    {:sparsity => Relation(to_idx) => :sparsity} ~ beta(BETA_PRIOR[1] + num_true, BETA_PRIOR[2] + num_entities^2)
end

@gen function dumb_split_merge_proposal(tr)
    prob_of_split = tr[:world => :num_relations => ()] > 1 ? 0.5 : 1.
    do_split ~ bernoulli(prob_of_split)
    if do_split
        {*} ~ dumb_split_proposal(tr)
    else
        {*} ~ dumb_merge_proposal(tr)
    end
end

@oupm_involution dumb_splitmerge_involution (old, fwd) to (new, bwd) begin
    do_split = @read(fwd[:do_split], :disc)
    @write(bwd[:do_split], !do_split, :disc)
    if do_split
        # println("Beginning SPLIT involution")
        from_idx = @read(fwd[:from_idx], :disc)
        to1 = @read(fwd[:to_idx1], :disc)
        to2 = @read(fwd[:to_idx2], :disc)
        @write(bwd[:from_idx1], to1, :disc)
        @write(bwd[:from_idx2], to2, :disc)
        @write(bwd[:to_idx], from_idx, :disc)

        # SPLIT
        moves = Tuple(
            Fact((Relation(from_idx), Entity(e1), Entity(e2)), 1) => nothing for e1=1:get_args(tr)[1], e2=1:get_args(tr)[1]
            if @read(old[:world => :num_facts => (Relation(from_idx), Entity(e1), Entity(e2)) => :is_true], :disc)
        )
        @split(Relation(from_idx), to1, to2, moves)
        @write(new[:world => :num_relations => () => :num], @read(old[:world => :num_relations => () => :num], :disc) + 1, :disc)

        # REFERENCES, VERB PRIOR, SPARSITY
        abst_rel = @convert_to_abstract(Relation(from_idx))
        if haskey(@read(old[:kernel => :counts], :disc).indices_per_entity, abst_rel)
            for idx in @read(old[:kernel => :counts], :disc).indices_per_entity[abst_rel]
                new_idx = @read(fwd[:to_lower => idx], :disc) ? to1 : to2
                @write(
                    new[:kernel => :facts => :sampled_facts => idx],
                    Fact((Relation(new_idx), Entity(get_retval(tr)[idx][1]), Entity(get_retval(tr)[idx][3])), 1),
                    :disc
                )
            end
            @copy(old[:world => :verb_prior => Relation(from_idx)], bwd[:verb_prior => Relation(from_idx)])
        end
        @copy(fwd[:verb_prior], new[:world => :verb_prior])
        @copy(fwd[:sparsity], new[:world => :sparsity])
        @copy(old[:world => :sparsity => Relation(from_idx)], bwd[:sparsity => Relation(from_idx)])
        
        # TRUTHINESS
        for e1=1:get_args(tr)[1], e2=1:get_args(tr)[1]
            if @read(old[:world => :num_facts => (Relation(from_idx), Entity(e1), Entity(e2)) => :is_true], :disc)
                t = @read(fwd[:truthiness => (e1, e2)], :disc)
                true1 = t == 1 || t == 3
                true2 = t == 2 || t == 3
            else
                true1 = false
                true2 = false
            end
            @write(new[:world => :num_facts => (Relation(to1), Entity(e1), Entity(e2)) => :is_true], true1, :disc)
            @write(new[:world => :num_facts => (Relation(to2), Entity(e1), Entity(e2)) => :is_true], true2, :disc)
        end
    else
        # println("Beginning MERGE involution")
        to_idx = @read(fwd[:to_idx], :disc)
        from1 = @read(fwd[:from_idx1], :disc)
        from2 = @read(fwd[:from_idx2], :disc)
        @write(bwd[:to_idx1], from1, :disc)
        @write(bwd[:to_idx2], from2, :disc)
        @write(bwd[:from_idx], to_idx, :disc)

        # MERGE
        moves = Tuple(
            Iterators.flatten(
                (
                    (
                        Fact((Relation(from1), Entity(e1), Entity(e2)), 1) => nothing
                        for e1=1:get_args(tr)[1], e2=1:get_args(tr)[1]
                        if @read(old[:world => :num_facts => (Relation(from1), Entity(e1), Entity(e2)) => :is_true], :disc)
                    ),
                    (
                        Fact((Relation(from2), Entity(e1), Entity(e2)), 1) => nothing
                        for e1=1:get_args(tr)[1], e2=1:get_args(tr)[1]
                        if @read(old[:world => :num_facts => (Relation(from2), Entity(e1), Entity(e2)) => :is_true], :disc)
                    )
                )
            )
        )
        @merge(Relation(to_idx), from1, from2, moves)
        @write(new[:world => :num_relations => () => :num], @read(old[:world => :num_relations => () => :num], :disc) - 1, :disc)

        # TO LOWER, DIRICHLET, SPARSITY
        abst_rel1 = @convert_to_abstract(Relation(from1))
        abst_rel2 = @convert_to_abstract(Relation(from2))
        if haskey(@read(old[:kernel => :counts], :disc).indices_per_entity, abst_rel1)
            for idx in @read(old[:kernel => :counts], :disc).indices_per_entity[abst_rel1]
                # println("$idx, ")
                @write(
                    new[:kernel => :facts => :sampled_facts => idx],
                    Fact((Relation(to_idx), Entity(get_retval(tr)[idx][1]), Entity(get_retval(tr)[idx][3])), 1),
                    :disc
                )
                @write(bwd[:to_lower => idx], true, :disc)
            end
            @copy(old[:world => :verb_prior => Relation(from1)], bwd[:verb_prior => Relation(from1)])
        end
        if haskey(@read(old[:kernel => :counts], :disc).indices_per_entity, abst_rel2)
            for idx in @read(old[:kernel => :counts], :disc).indices_per_entity[abst_rel2]
                # print("$idx, ")
                @write(
                    new[:kernel => :facts => :sampled_facts => idx],
                    Fact((Relation(to_idx), Entity(get_retval(tr)[idx][1]), Entity(get_retval(tr)[idx][3])), 1),
                    :disc
                )
                @write(bwd[:to_lower => idx], false, :disc)
            end
            @copy(old[:world => :verb_prior => Relation(from2)], bwd[:verb_prior => Relation(from2)])
        end
        @copy(fwd[:verb_prior], new[:world => :verb_prior])
        @copy(fwd[:sparsity], new[:world => :sparsity])
        @copy(old[:world => :sparsity => Relation(from1)], bwd[:sparsity => Relation(from1)])
        @copy(old[:world => :sparsity => Relation(from2)], bwd[:sparsity => Relation(from2)])

        # TRUTHINESS
        for e1=1:get_args(tr)[1], e2=1:get_args(tr)[1]
            old1_true = @read(old[:world => :num_facts => (Relation(from1), Entity(e1), Entity(e2)) => :is_true], :disc)
            old2_true = @read(old[:world => :num_facts => (Relation(from2), Entity(e1), Entity(e2)) => :is_true], :disc)
            @write(new[:world => :num_facts => (Relation(to_idx), Entity(e1), Entity(e2)) => :is_true], old1_true || old2_true, :disc)

            if old1_true || old2_true
                if old1_true && old2_true
                    val = 3
                elseif old1_true
                    val = 1
                else
                    val = 2
                end
                @write(bwd[:truthiness => (e1, e2)], val, :disc)
            end
        end
    end
end
dumb_splitmerge_kernel = OUPMMHKernel(dumb_split_merge_proposal, (), dumb_splitmerge_involution)