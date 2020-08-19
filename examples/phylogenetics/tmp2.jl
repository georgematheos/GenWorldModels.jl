NUM_WORDS = 5
α_val = 1.
α = fill(α_val, NUM_WORDS)

@type Object
@gen num_objects(::World, ::Tuple{}) = {:num} ~ poisson(7)
@gen (static, diffs) function prior_for_object(world::World, object::Object)
    prior ~ dirichlet(α)
    return prior
end
@gen (static, diffs) function sample_mention(world::World, object::Object)
    prior ~ prior_for_object[object]
    mention ~ categorical(prior)
    return mention
end
@gen (static, diffs) function entity_mention_model(world, num_samples)
    num_objs ~ lookup_or_generate(world[:num_objects][()])
    abstract_objs ~ Map(lookup_or_generate)(mgfcall_map(world[:abstract], 1:num_objs))
    sampled_objs = Map(uniform_from_set)(fill(Set(abstract_objs), num_samples))
    sampled_mentions = Map(sample_mention)(fill(world, num_samples), sampled_objs)
    counts ~ get_emm_counts(sampled_objs, sampled_mentions)
    return sampled_mentions
end

struct GetEmmCounts <: CustomUpdateGF end
get_emm_counts = GetEmmCounts()
Gen.has_argument_grads(::GetEmmCounts) = false
function Gen.apply_with_state(::GetEmmCounts, (num_entities, num_mentions, entities, mentions))
    num_m_per_e = PersistentHashMap([entity => PersistentVector(zeros(num_mentions)) for _=1:num_entities])
    m_per_e = PersistentHashMap([entity => PersistentSet() for _=1:num_entities])

    for (entity, mention) in zip(entities, mentions)
        num_m_per_e = assoc(num_m_per_e, entity,
            assoc(num_m_per_e[entity], mention, num_m_per_e[entity][mention] + 1)
        )
        m_per_e = assoc(m_per_e, entity, push(m_per_e[entity], mention))
    end

    return ((num_m_per_e, m_per_e), (num_m_per_e, m_per_e, entities))
end
function Gen.update_with_state(
    ::GetEMMCounts,
    (num_m_per_e, m_per_e, old_entities),
    (num_entities, num_mentions, entities, mentions),
    (_, _, entitydiff, _)::Tuple{<:Gen.Diff, NoChange, <:Gen.VectorDiff, NoChange}
)
    @assert entitydiff.old_length == entitydiff.new_length
    for (idx, diff) in entitydiff.updated
        diff === NoChange() && continue;
        old_ent = old_entities[idx]
        new_ent = entities[idx]
        old_ent == new_ent && continue;
        mention = mentions[idx]
        old_cnt = num_m_per_e[old_ent][mention]
        num_m_per_e = assoc(num_m_per_e, old_ent,
            assoc(num_m_per_e[old_ent], mention, num_m_per_e[old_ent][mention] - 1)
        )
        num_m_per_e = assoc(num_m_per_e, new_ent,
            assoc(num_m_per_e[new_ent], mention, num_m_per_e[new_ent][mention] + 1)
        )
        if old_cnt == 1
            m_per_e = assoc(m_per_e, old_ent, dissoc(m_per_e[old_ent], mention))
        end
        m_per_e = assoc(m_per_e, new_ent, push(m_per_e[new_ent], mention))
    end

    return ((num_m_per_e, m_per_e, entities), (num_m_per_e, m_per_e), UnknownChange())
end



@gen function sdds_proposal(tr, counts, mentions_per_entity)
    do_smart_split_dumb_merge ~ bernoulli(0.5)
    if do_smart_split_dumb_merge
        {*} ~ smart_split_dumb_merge_proposal(tr, counts, mentions_per_entity)
    else
        {*} ~ smart_split_dumb_merge_proposal(tr, counts, mentions_per_entity)
    end
end

@gen function smart_split_dumb_merge_proposal(tr, counts, mentions_per_entity)
    do_split ~ bernoulli(0.5)
    if do_split
        {*} ~ smart_split_proposal(tr, counts, mentions_per_entity)
    else
        {*} ~ dumb_merge_proposal(tr, counts, mentions_per_entity)
    end
end

@gen function smart_split_proposal(tr)
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
    for mention in mentions_per_entity[to_split]
        p1 = (α_val + num_1_per_men[mention])/(tot_α + tot_num_1)
        p2 = (α_val + num_2_per_men[mention])/(tot_α + tot_num_2)
        to_idx_1 = {:to_idx_1 => mention} ~ bernoulli(p1 / (p1 + p2))

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

    else

    end
end