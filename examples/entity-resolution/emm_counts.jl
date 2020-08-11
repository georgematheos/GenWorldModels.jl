struct GetEmmCounts <: CustomUpdateGF end
get_emm_counts = GetEmmCounts()
Gen.has_argument_grads(::GetEmmCounts) = false
function Gen.apply_with_state(::GetEmmCounts, (num_entities, num_mentions, entities, mentions))
    num_m_per_e = PersistentHashMap([entity => PersistentVector(zeros(num_mentions)) for _=1:num_entities])
    indices_per_e = PersistentHashMap([entity => PersistentSet() for _=1:num_entities])

    for (i, (entity, mention)) in enumerate(zip(entities, mentions))
        num_m_per_e = assoc(num_m_per_e, entity,
            assoc(num_m_per_e[entity], mention, num_m_per_e[entity][mention] + 1)
        )
        indices_per_e = assoc(m_per_e, entity, push(m_per_e[entity], i))
    end

    return ((num_m_per_e, m_per_e), (num_m_per_e, m_per_e, entities))
end
function Gen.update_with_state(
    ::GetEMMCounts,
    (num_m_per_e, indices_per_e, old_entities),
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
        indices_per_e = assoc(indices_per_e, old_ent, dissoc(m_per_e[old_ent], idx))
        indices_per_e = assoc(indices_per_e, new_ent, push(m_per_e[new_ent], idx))
    end

    return ((num_m_per_e, indices_per_e, entities), (num_m_per_e, indices_per_e), UnknownChange())
end