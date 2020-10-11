struct ConvertingVectorValueChoiceMap <: Gen.AddressTree{Value}
    world::World
    vector::AbstractVector
end
function Gen.get_subtree(c::ConvertingVectorValueChoiceMap, a::Int)
    if a <= length(c.vector)
        fact = c.vector[a]
        Value(conv_fact_to_conc(fact, c.world))
    else
        EmptyAddressTree()
    end
end
Gen.get_subtree(::ConvertingVectorValueChoiceMap, ::Pair) = EmptyAddressTree()
Gen.get_subtrees_shallow(c::ConvertingVectorValueChoiceMap) = ((i, get_subtree(c, i)) for i=1:length(c.vector))

struct MapUniformFactTrace <: Gen.Trace
    world::World
    factset::AbstractSet
    samples::PersistentVector
    counts::PersistentHashMap
    score::Float64
end
Gen.get_retval(tr::MapUniformFactTrace) = tr.samples
Gen.get_choices(tr::MapUniformFactTrace) = ConvertingVectorValueChoiceMap(tr.world, tr.samples)
Gen.get_score(tr::MapUniformFactTrace) = tr.score
Gen.project(tr::MapUniformFactTrace, ::EmptyAddressTree) = 0.
Gen.get_args(tr::MapUniformFactTrace) = (length(tr.samples), tr.factset)

struct MapUniformFact <: Gen.GenerativeFunction{Any, MapUniformFactTrace} end
map_uniform_fact = MapUniformFact();

conv_fact_to_abst(fact, world) = Fact(GenWorldModels.convert_to_abstract(world, fact.rel), fact.ent1, fact.ent2)
conv_fact_to_conc(fact, world) = Fact(GenWorldModels.convert_to_concrete(world, fact.rel), fact.ent1, fact.ent2)

function Gen.generate(::MapUniformFact, (world, num, factset)::Tuple{<:World, <:Integer, <:AbstractSet}, constraints::ChoiceMap)
    samples = []
    counts = PersistentHashMap()
    num_constrained = 0
    is_impossible = false
    for i=1:num
        constr = get_subtree(constraints, i)
        if !isempty(constr)
            newval = get_value(constr)
            newval = conv_fact_to_abst(newval, world)
            num_constrained += 1
            push!(samples, newval)
            if !(newval in factset)
                is_impossible = true
            end
        else
            push!(samples, uniform_choice(factset))
        end
        if haskey(counts, samples[i])
            counts = assoc(counts, samples[i], counts[samples[i]] + 1)
        else
            counts = assoc(counts, samples[i], 1)
        end
    end
    score = is_impossible ? -Inf : -log(length(factset))*num
    weight = is_impossible ? -Inf : -log(length(factset))*num_constrained
    return (MapUniformFactTrace(world, factset, PersistentVector(samples), counts, score), weight)
end

function Gen.update(
    tr::MapUniformFactTrace,
    (world, num, factset)::Tuple,
    (_, _, setdiff)::Tuple{<:Diff, NoChange, <:Union{<:SetDiff, NoChange}},
    constraints::UpdateSpec,
    eca::Selection
)
    samples = tr.samples
    discard = choicemap()
    updated = Dict{Int, Diff}()
    counts = tr.counts

    is_impossible = false
    for (i, subtree) in get_subtrees_shallow(constraints)
        newval = conv_fact_to_abst(get_value(subtree), world)
        if !(newval in factset)
            is_impossible = true
        end
        discard[i] = samples[i]
        
        newcount = counts[samples[i]] - 1
        if newcount == 0
            counts = dissoc(counts, samples[i])
        else
            counts = assoc(counts, samples[i], newcount)
        end

        samples = assoc(samples, i, newval)
        updated[i] = UnknownChange()

        if haskey(counts, newval)
            counts = assoc(counts, newval, counts[newval] + 1)
        else
            counts = assoc(counts, newval, 1)
        end
    end

    if setdiff isa SetDiff
        for del in setdiff.deleted
            if haskey(counts, del)
                is_impossible = true
            end
        end
    end

    newscore = is_impossible ? -Inf : num * -log(length(factset))
    new_tr = MapUniformFactTrace(world, factset, samples, counts, newscore)
    return (new_tr, newscore - get_score(tr), VectorDiff(num, num, updated), discard)
end