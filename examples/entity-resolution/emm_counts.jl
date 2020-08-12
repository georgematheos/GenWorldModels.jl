"""
    EntityMentionCounts

Stores the number of times each mention was used to refer to each entity,
and which indices contain each entity in the list of entities.
"""
struct EntityMentionCounts
    num_mentions_per_entity::PersistentHashMap{Any, PersistentVector{Int}}
    indices_per_entity::PersistentHashMap{Any, PersistentSet{Int}}
end
function EntityMentionCounts()
    nmpe = PersistentHashMap{Any, PersistentVector{Int}}()
    ipe = PersistentHashMap{Any, PersistentSet{Int}}()
    EntityMentionCounts(nmpe, ipe)
end
function add_entity_if_needed(n_m_per_e, idx_per_e, entity, num_mentions)
    if !haskey(n_m_per_e, entity)
        n_m_per_e = assoc(n_m_per_e, entity, PersistentVector(zeros(Int, num_mentions)))
    end
    if !haskey(idx_per_e, entity)
        idx_per_e = assoc(idx_per_e, entity, PersistentSet{Int}())
    end

    (n_m_per_e, idx_per_e)
end
function note_entity_mention_at_index(counts, idx, entity, mention, num_mentions)
    n_m_per_e = counts.num_mentions_per_entity
    idx_per_e = counts.indices_per_entity
    n_m_per_e, idx_per_e = add_entity_if_needed(n_m_per_e, idx_per_e, entity, num_mentions)
    n_m_per_e = assoc(n_m_per_e, entity, assoc(n_m_per_e[entity], mention, n_m_per_e[entity][mention] + 1))
    idx_per_e = assoc(idx_per_e, entity, push(idx_per_e[entity], idx))

    EntityMentionCounts(n_m_per_e, idx_per_e)
end
function note_entity_changed_at_index(counts, idx, old_entity, new_entity, mention, num_mentions)
    n_m_per_e = counts.num_mentions_per_entity
    idx_per_e = counts.indices_per_entity
    n_m_per_e, idx_per_e = add_entity_if_needed(n_m_per_e, idx_per_e, new_entity, num_mentions)
    n_m_per_e = assoc(n_m_per_e, new_entity, assoc(n_m_per_e[new_entity], mention, n_m_per_e[new_entity][mention] + 1))
    idx_per_e = assoc(idx_per_e, new_entity, push(idx_per_e[new_entity], idx))


    new_idxset = disj(idx_per_e[old_entity], idx)
    if isempty(new_idxset)
        n_m_per_e = dissoc(n_m_per_e, old_entity)
        idx_per_e = dissoc(idx_per_e, old_entity)
    else
        new_count = n_m_per_e[old_entity][mention] - 1
        n_m_per_e = assoc(n_m_per_e, old_entity, assoc(n_m_per_e[old_entity], mention, new_count))
        idx_per_e = assoc(idx_per_e, old_entity, new_idxset)
    end

    return EntityMentionCounts(n_m_per_e, idx_per_e)
end

"""
    get_entity_mention_counts(num_mentions, entities, mentions)

Given a vector of `entities` and a vector of `mentions` (where entities
may be any object, and mentions are integers between `1:num_mentions`),
counts the number of times each mention is used to refer to each entity,
and which indices contain each entity.
"""
struct GetEntityMentionCounts <: CustomUpdateGF{EntityMentionCounts, Tuple{EntityMentionCounts, <:Vector}} end
get_entity_mention_counts = GetEntityMentionCounts()
Gen.has_argument_grads(::GetEntityMentionCounts) = false

function Gen.apply_with_state(::GetEntityMentionCounts, (num_mentions, entities, mentions))
    counts = EntityMentionCounts()

    for (i, (entity, mention)) in enumerate(zip(entities, mentions))
        counts = note_entity_mention_at_index(counts, i, entity, mention, num_mentions)
    end

    return (counts, (counts, entities))
end

# TODO: handle update where mentions also change
function Gen.update_with_state(
    ::GetEntityMentionCounts,
    (old_counts, old_entities),
    (num_mentions, entities, mentions),
    (_, entitydiff, _)::Tuple{NoChange, <:Gen.VectorDiff, NoChange}
)
    @assert entitydiff.old_length == entitydiff.new_length

    counts = old_counts
    for (idx, diff) in entitydiff.updated
        diff === NoChange() && continue;
        old_ent = old_entities[idx]
        new_ent = entities[idx]
        old_ent == new_ent && continue;
        mention = mentions[idx]
        counts = note_entity_changed_at_index(counts, idx, old_ent, new_ent, mention, num_mentions)
    end

    return ((counts, entities), counts, UnknownChange())
end