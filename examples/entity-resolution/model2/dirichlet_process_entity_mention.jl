# TODO: I should rename this to something like `dirichlet_process_entity_mention`

struct VectorChoiceMap{T} <: Gen.AddressTree{Value}
    v::AbstractVector{T}
end
Gen.get_subtree(c::VectorChoiceMap, i::Int) = Gen.Value(c.v[i])
Gen.get_subtree(::VectorChoiceMap, ::Pair) = EmptyAddressTree()
Gen.get_subtrees_shallow(c::VectorChoiceMap) = ((i, Gen.Value(val)) for (i, val) in enumerate(c.v))

struct DirichletProcessEntityMentionTrace{EntityType} <: Gen.Trace
    args::Tuple{AbstractVector{EntityType}, AbstractVector{<:Real}}
    counts::PersistentHashMap{EntityType, PersistentVector{Int}}
    mentions::AbstractVector{Int}
    entity_to_indices::PersistentHashMap{EntityType, PersistentSet{Int}}
    score::Float64
end
Gen.get_gen_fn(::DirichletProcessEntityMentionTrace) = DirichletProcessEntityMention()
Gen.get_args(tr::DirichletProcessEntityMentionTrace) = tr.args
Gen.get_retval(tr::DirichletProcessEntityMentionTrace) = tr.mentions
Gen.get_score(tr::DirichletProcessEntityMentionTrace) = tr.score
Gen.get_choices(tr::DirichletProcessEntityMentionTrace) = VectorChoiceMap(tr.mentions)
Gen.project(::DirichletProcessEntityMentionTrace, ::EmptyAddressTree) = 0.

_get_score(counts, α) = sum(logbeta(α + count) for (_, count) in counts) - (length(counts) * logbeta(α))
function DirichletProcessEntityMentionTrace{EntityType}(args, counts, mentions, ent_to_indices) where {EntityType}
    α = args[2]
    DirichletProcessEntityMentionTrace{EntityType}(args, counts, mentions, ent_to_indices, _get_score(counts, α))
end

struct DirichletProcessEntityMention <: Gen.GenerativeFunction{
    AbstractVector{Int},
    DirichletProcessEntityMentionTrace
} end

"""
    dirichlet_process_entity_mention(entity_list, α)

Given a list of (potentially non-unique) entities, samples a mention for each appearance
of each entity according to a dirichlet process with parameter α.
"""
dirichlet_process_entity_mention = DirichletProcessEntityMention()

normalize(v) = v./sum(v)
function sample_mention_and_count!(counts, unconstrained_counts, entity, α, ::EmptyAddressTree)
    mention = categorical(normalize(counts[entity] + α))
    counts[entity][mention] += 1
    unconstrained_counts[entity][mention] += 1
    mention
end 
function sample_mention_and_count!(counts, _, entity, _, constraint::Value)
    mention = get_value(constraint)
    counts[entity][mention] += 1
    mention
end

function to_persistent(d::Dict{T, Vector{Int}}) where {T}
    phm = PersistentHashMap{T, PersistentVector{Int}}()
    for (k, v) in d
        phm = assoc(phm, k, PersistentVector{Int}(v))
    end
    phm
end

function Gen.generate(
    ::DirichletProcessEntityMention,
    args::Tuple{AbstractVector{EntityType}, Vector{<:Real}},
    constraints::ChoiceMap
) where {EntityType}
    (entities, α) = args
    unique_entities = unique(entities)
    num_mentions = length(α)
    counts = Dict{EntityType, Vector{Int}}((ent => zeros(num_mentions) for ent in unique_entities))
    unconstrained_counts = Dict{EntityType, Vector{Int}}((ent => zeros(num_mentions) for ent in unique_entities))
    mentions = Vector{Int}(undef, length(entities))
    ent_to_indices = PersistentHashMap{EntityType, PersistentSet{Int}}()
    for (i, entity) in enumerate(entities)
        if haskey(ent_to_indices, entity)
            ent_to_indices = assoc(ent_to_indices, entity, push(ent_to_indices[entity], i))
        else
            ent_to_indices = assoc(ent_to_indices, entity, PersistentSet{Int}([i]))
        end

        mentions[i] = sample_mention_and_count!(counts, unconstrained_counts, entity, α, get_submap(constraints, i))
    end
    tr = DirichletProcessEntityMentionTrace{EntityType}(args, to_persistent(counts), mentions, ent_to_indices)
    weight = get_score(tr) - _get_score(unconstrained_counts, α)
    (tr, weight)
end

# this update method is for the case where we change the entities, but the α and the mentions remain the same
function Gen.update(
    tr::DirichletProcessEntityMentionTrace{EntityType},
    args::Tuple{AbstractVector{EntityType}, Vector{<:Real}},
    argdiffs::Tuple{VectorDiff, NoChange},
    constraints::EmptyAddressTree,
    ext_const_addrs::Selection
) where {EntityType}
    diff = argdiffs[1]
    if diff.new_length !== diff.prev_length
        error("Not implemented: the situation where the length of an entity-mention vector is changing.")
    end
    old_entities = get_args(tr)[1]
    new_entities, α = args
    num_mentions = length(α)
    ent_to_indices = tr.entity_to_indices

    new_ent_set = Set()
    old_ent_set = Set()
    new_counts = tr.counts
    for (idx, diff) in diff.updated
        diff === NoChange() && continue;

        old_entity = old_entities[idx]
        new_entity = new_entities[idx]
        mention = get_retval(tr)[idx]
        old_entity === new_entity && continue;

        ent_to_indices = assoc(ent_to_indices, old_entity, disj(ent_to_indices[old_entity], idx))
        if haskey(ent_to_indices, new_entity)
            ent_to_indices = assoc(ent_to_indices, new_entity, push(ent_to_indices[new_entity], idx))
        else
            ent_to_indices = assoc(ent_to_indices, new_entity, PersistentSet{Int}([idx]))
        end

        old_old_ent_count = new_counts[old_entity]
        new_old_ent_count = assoc(old_old_ent_count, mention, old_old_ent_count[mention] - 1)
        old_new_ent_count = haskey(new_counts, new_entity) ? new_counts[new_entity] : PersistentVector{Int}(zeros(num_mentions))
        new_new_ent_count = assoc(old_new_ent_count, mention, old_new_ent_count[mention] + 1)
        new_counts = assoc(new_counts, old_entity, new_old_ent_count)
        new_counts = assoc(new_counts, new_entity, new_new_ent_count)

        push!(new_ent_set, new_entity)
        push!(old_ent_set, old_entity)
    end

    # remove any entities from the dictionary if there are no references to them; update the score
    Δlogprob = 0.
    old_counts = tr.counts
    for entity in old_ent_set
        Δlogprob -= logbeta(α + old_counts[entity])
        Δlogprob += logbeta(α + new_counts[entity])
        if sum(new_counts[entity]) === 0
            new_counts = dissoc(new_counts, entity)
            ent_to_indices = dissoc(ent_to_indices, entity)
        end
    end
    for entity in new_ent_set
        if !(entity in old_ent_set)
            Δlogprob -= logbeta(α + get(old_counts, entity, zeros(num_mentions)))
            Δlogprob += logbeta(α + new_counts[entity])
        end
    end
    new_tr = DirichletProcessEntityMentionTrace{EntityType}(args, new_counts, get_retval(tr), ent_to_indices, get_score(tr) + Δlogprob)
    (new_tr, Δlogprob, NoChange(), EmptyAddressTree())
end

function Base.getindex(tr::DirichletProcessEntityMentionTrace, addr)
    if addr == :indices_per_entity
        tr.entity_to_indices
    elseif addr == :counts
        tr.counts
    else
        get_choices(tr)[addr]
    end
end