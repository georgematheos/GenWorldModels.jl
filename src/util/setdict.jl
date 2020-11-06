# Would it be better to name this something relating to preimages?
# Currently I am only using it for preimage tracking.

"""
    SetDict

A persistent mapping from keys to sets.  Uses the read interface
of a `Dict{KeyType, Set}`.  Add `v` to the set for key `k`
using `FunctionalCollections.assoc(setdict, k, v)`, and remove
this using `FunctionalCollections.dissoc(setdict, k, v)`.
"""
struct SetDict{K, V} <: AbstractDict{K, PersistentSet{V}}
    map::PersistentHashMap{K, PersistentSet{V}}
end
SetDict{K, V}() where {K, V} = SetDict(PersistentHashMap{K, PersistentSet{V}}())
Base.haskey(sm::SetDict, k) = haskey(sm.map, k)
Base.iterate(sm::SetDict) = iterate(sm.map)
Base.iterate(sm::SetDict, st) = iterate(sm.map, st)
Base.getindex(sm::SetDict, i) = sm.map[i]

function FunctionalCollections.assoc(sm::SetDict, k, v)
    if haskey(sm.map, k)
        SetDict(assoc(sm.map, k, push(sm.map[k], v)))
    else
        SetDict(assoc(sm.map, k, PersistentSet(v)))
    end
end
function FunctionalCollections.dissoc(sm::SetDict, k, v)
    @assert (v in sm.map[k]) "$v is not in the setmap at key $k"
    if length(sm.map[k]) == 1
        SetDict(dissoc(sm.map, k))
    else
        SetDict(assoc(sm.map, k, disj(sm.map[k], v)))
    end
end

"""
    item_to_indices(vect)

Return a `SetDict` mapping the values in `vect` to the set of all indices
containing those values.
"""
item_to_indices(vect::AbstractVector{T}) where {T} = item_to_indices(T, vect)
item_to_indices(vect::Tuple{Vararg{T}}) where {T} = item_to_indices(T, vect)
item_to_indices(vect::Tuple) = item_to_indices(Any, vect)
function item_to_indices(T, vect)
    sm = SetDict{T, Int}()
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
    sm = SetDict{V, K}()
    for (k, v) in dict
        sm = assoc(sm, v, k)
    end
    sm
end