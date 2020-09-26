### A Pluto.jl notebook ###
# v0.11.12

using Markdown
using InteractiveUtils

# ╔═╡ e1b53ca6-f913-11ea-17a9-55c29d694e80
using Pkg; Pkg.activate(".")

# ╔═╡ f5c96eec-f913-11ea-32f9-f57a36861a4f
begin
	using Gen
	using FunctionalCollections
end

# ╔═╡ eccb66d0-f913-11ea-3d29-212ee68bc699
begin
# TODO: I should rename this to something like `dirichlet_process_entity_mention`

struct VectorChoiceMap{T} <: Gen.AddressTree{Value}
    v::AbstractVector{T}
end
Gen.get_subtree(c::VectorChoiceMap, i::Int) = Gen.Value(c.v[i])
Gen.get_subtrees_shallow(c::VectorChoiceMap) = ((i, Gen.Value(val)) for (i, val) in enumerate(c.v))
end

# ╔═╡ 4da44e02-f914-11ea-26e2-ffc770fdc197
begin
struct DirichletProcessEntityMentionTrace{EntityType} <: Gen.Trace
    args::Tuple{AbstractVector{EntityType}, AbstractVector{<:Real}}
    counts::PersistentHashMap{EntityType, PersistentVector{Int}}
    mentions::AbstractVector{Int}
    score::Float64
end
Gen.get_gen_fn(::DirichletProcessEntityMentionTrace) = DirichletProcessEntityMention()
Gen.get_args(tr::DirichletProcessEntityMentionTrace) = tr.args
Gen.get_retval(tr::DirichletProcessEntityMentionTrace) = tr.mentions
Gen.get_score(tr::DirichletProcessEntityMentionTrace) = tr.score
Gen.get_choices(tr::DirichletProcessEntityMentionTrace) = VectorChoiceMap(v)
Gen.project(::DirichletProcessEntityMentionTrace, ::EmptyAddressTree) = 0.


function DirichletProcessEntityMentionTrace{EntityType}(args, counts, mentions) where {EntityType}
    α = args[2]
    score = sum(logbeta(α + count) for (_, count) in counts) - (length(counts) * logbeta(α))
    DirichletProcessEntityMentionTrace{EntityType}(args, counts, mentions, score)
end
	
struct DirichletProcessEntityMention <: Gen.GenerativeFunction{
    AbstractVector{Int},
    DirichletProcessEntityMentionTrace
} end
	
dirichlet_process_entity_mention = DirichletProcessEntityMention()
end

# ╔═╡ 4da4ffb6-f914-11ea-3ccf-531359b01a4f
begin
function sample(counts, entity, α, constraint::EmptyAddressTree)
    mention = categorical(counts[entity] + α)
    (mention, 0.)
end
function sample(counts, entity, α, constraint::Value)
    mention = get_value(constraint)
    pdf = logpdf(categorical, mention, counts[entity] + α)
    (mention, pdf)
end

function to_persistent(d::Dict{T, Vector{Int}}) where {T}
    phm = PersistentHashMap{T, PersistentVector{Int}}()
    for v in d
        phm = assoc(phm, PersistentVector{Int}(v))
    end
    phm
end
end

# ╔═╡ 4daf25c0-f914-11ea-35e5-35f97e89c8c4
function Gen.generate(
    ::DirichletProcessEntityMention,
    args::Tuple{AbstractVector{EntityType}, Vector{<:Real}},
    constraints::ChoiceMap
) where {EntityType}
    (entities, α) = args
    num_mentions = length(α)
    counts = Dict{EntityType, Int}()
    mentions = Vector{Int}(undef, length(entities))
    weight = 0.
    for (i, entity) in enumerate(entities)
        if !haskey(counts, entity)
            counts[entity] = zeros(num_mentions)
        end
        mention, Δweight = sample(counts, entity, α, get_submap(constraints, i))
        counts[entity][mention] += 1
        mentions[i] = mention
        weight += Δweight
    end
    (
        DirichletProcessEntityMentionTrace{EntityType}(args, to_persistent(counts), mentions),
        weight
    )
end

# ╔═╡ 4dafa1e4-f914-11ea-37a9-f991f510045c
# this update method is for the case where we change the entities, but the α and the mentions remain the same

function Gen.update(
    tr::DirichletProcessEntityMentionTrace{EntityType},
    args::Tuple{AbstractVector{EntityType}, Vector{<:Real}},
    argdiffs::Tuple{VectorDiff, NoChange},
    constraints::EmptyAddressTree,
    ext_const_addrs::Selection
) where {EntityType}
    diff = argdiffs[1]
    if diff.new_length !== diff.old_length
        error("Not implemented: the situation where the length of an entity-mention vector is changing.")
    end
    old_entities = get_args(tr)[1]
    new_entities, α = args
    num_mentions = length(α)

    new_entities = Set()
    old_entities = Set()
    new_counts = tr.counts
    for (idx, diff) in diff.updated
        diff === NoChange() && continue;

        old_entity = old_entities[idx]
        new_entity = new_entities[idx]
        mention = get_retval(tr)[idx]
        old_entity === new_entity && continue;

        old_old_ent_count = new_counts[old_entity]
        new_old_ent_count = assoc(old_old_ent_count, mention, old_old_ent_count[mention] - 1)
        old_new_ent_count = haskey(new_counts, new_entity) ? new_counts[new_entity] : PersistentVector{Int}(zeros(num_mentions))
        new_new_ent_count = assoc(old_new_ent_count, mention, old_new_ent_count[mention] + 1)
        new_counts = assoc(new_counts, old_entity, new_old_ent_count)
        new_counts = assoc(new_counts, new_entity, new_new_ent_count)

        push!(new_entities, new_entity)
        push!(old_entities, old_entity)
    end

    # remove any entities from the dictionary if there are no references to them; update the score
    Δlogprob = 0.
    for entity in old_entities
        if sum(new_counts[entity]) === 0
            new_counts = dissoc(new_counts, entity)
        end
        Δlogprob -= logbeta(α + old_counts[entity])
    end
    for entity in new_entities
        Δlogprob += logbeta(α + new_counts[entity])
    end

    new_tr = DirichletProcessEntityMentionTrace{EntityType}(args, new_counts, get_retval(tr), get_score(tr) + Δlogprob)
    (new_tr, Δlogprob, NoChange(), EmptyAddressTree())
end

# ╔═╡ 1c327880-f914-11ea-1798-5bcd2c2ce174
generate(dirichlet_process_entity_mention, (["a", "b", "b"], [0.2, 0.2]))

# ╔═╡ Cell order:
# ╠═e1b53ca6-f913-11ea-17a9-55c29d694e80
# ╠═f5c96eec-f913-11ea-32f9-f57a36861a4f
# ╠═eccb66d0-f913-11ea-3d29-212ee68bc699
# ╠═4da44e02-f914-11ea-26e2-ffc770fdc197
# ╠═4da4ffb6-f914-11ea-3ccf-531359b01a4f
# ╠═4daf25c0-f914-11ea-35e5-35f97e89c8c4
# ╠═4dafa1e4-f914-11ea-37a9-f991f510045c
# ╠═1c327880-f914-11ea-1798-5bcd2c2ce174
