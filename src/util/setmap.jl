"""
    SetMap

A persistent mapping from keys to sets.  Uses the read interface
of a `Dict{KeyType, Set}`.  Add `v` to the set for key `k`
using `FunctionalCollections.assoc(setmap, k, v)`, and remove
this using `FunctionalCollections.dissoc(setmap, k, v)`.
"""
struct SetMap{K, V} <: AbstractDict{K, PersistentSet{V}}
    map::PersistentHashMap{K, PersistentSet{V}}
end
Base.haskey(sm::SetMap, k) = haskey(sm.map, k)
Base.iterate(sm::SetMap) = iterate(sm.map)
Base.iterate(sm::SetMap, st) = iterate(sm.map, st)
Base.getindex(sm::SetMap, i) = sm.map[i]

function FunctionalCollections.assoc(sm::SetMap, k, v)
    if haskey(sm.map, k)
        SetMap(assoc(sm.map, k, push(sm.map[k], v)))
    else
        SetMap(assoc(sm.map, k, PersistentSet(v)))
    end
end
function FunctionalCollections.dissoc(sm::SetMap, k, v)
    @assert (v in sm.map[k]) "$v is not in the setmap at key $k"
    if length(sm.map[k]) == 1
        SetMap(dissoc(sm.map, k))
    else
        SetMap(assoc(sm.map, k, disj(sm.map[k], v)))
    end
end

"""
    item_to_indices(vect)

Return a `SetMap` mapping the values in `vect` to the set of all indices
containing those values.
"""
function item_to_indices(vect::AbstractVector{T}) where {T}
    sm = SetMap{T, Int}
    for (i, v) in enumerate(vect)
        sm = assoc(sm, v, i)
    end
    sm
end

"""
    vals_to_keys(dict)

Return a `SetMap` mapping the values in `dict` to the set of all keys
which map to that value.
"""
function vals_to_keys(dict::AbstractDict{K, V}) where {K, V}
    sm = SetMap{T, Int}
    for (k, v) in dict
        sm = assoc(sm, v, k)
    end
    sm
end