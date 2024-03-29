struct UniqueValueSetState
    vals::PersistentSet
    dict::AbstractDict
end

struct UniqueValueSet <: Gen.CustomUpdateGF{PersistentSet, UniqueValueSetState} end

"""
    values ~ unique_value_set(dictionary)

Returns a set of all the values in the dictionary, given
that all the values are known to be unique.
"""
unique_value_set = UniqueValueSet()

function Gen.apply_with_state(::UniqueValueSet, (dict,)::Tuple{AbstractDict{K, V}}) where {K, V}
    vals = PersistentSet{V}(values(dict))
    (vals, UniqueValueSetState(vals, dict))
end
function Gen.update_with_state(::UniqueValueSet, st, (dict,), (diff,)::Tuple{DictDiff})
    set = st.vals
    added = Set()
    deleted = Set()
    for key in diff.deleted
        set = disj(set, st.dict[key])
        push!(deleted, st.dict[key])
    end
    for (key, diff) in diff.updated
        (diff === NoChange() || st.dict[key] == dict[key]) && continue
        set = disj(set, st.dict[key])
        push!(deleted, st.dict[key])
        set = push(set, dict[key])
        push!(added, dict[key])
    end
    for (k, newval) in diff.added
        set = push(set, newval)
        push!(added, newval)
    end

    added_and_deleted = intersect(added, deleted)
    setdiff!(added, added_and_deleted)
    setdiff!(deleted, added_and_deleted)

    (UniqueValueSetState(set, dict), set, SetDiff(added, deleted))
end