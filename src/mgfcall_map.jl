export mgfcall_map, mgfcall_setmap

"""
    mgfcall_map(memoized_gen_function, keys::AbstractVector)

Returns an abstract vector containing the MGF call of the given MGF with each key in `keys`.
This function is useful since it automates some diff propagation; it intuitively functions equivalently to writing
    
    [mgf[key] for key in keys]
"""
function mgfcall_map(mgf::MemoizedGenerativeFunction, keys)
    [mgf[key] for key in keys]
end

function mgfcall_map(mgf::Diffed{<:MemoizedGenerativeFunction}, keys::Diffed{<:Any, NoChange})
    k = strip_diff(keys)
    l = length(k)
    mgfcall_map(mgf, Diffed(k, VectorDiff(l, l, Dict())))
end
function mgfcall_map(mgf::Diffed{<:MemoizedGenerativeFunction}, keys::Diffed{<:Any, VectorDiff})
    # TODO: for large `keys`, it may be faster to iterate over all diffed items, rather than all keys
    diffed_vals = [mgf[key] for key in strip_diff(keys)]
    diffs = map(get_diff, diffed_vals)
    vals = map(strip_diff, diffed_vals)
    vdiff = get_diff(keys)

    changed = Dict{Int, Diff}()
    for (i, diff) in enumerate(diffs)
        if diff !== NoChange()
            changed[i] = diff
        end
    end
    for (i, diff) in vdiff.updated
        changed[i] = KeyChangedDiff(diff)
    end

    Diffed(vals, VectorDiff(vdiff.new_length, vdiff.prev_length, changed))
end
function mgfcall_map(mgf::Diffed{<:MemoizedGenerativeFunction}, keys::Diffed{<:Any})
    Diffed(mgfcall_map(strip_diff(mgf), strip_diff(keys)), UnknownChange())
end

mgfcall_map(mgf::MemoizedGenerativeFunction, keys::Diffed) = mgfcall_map(Diffed(mgf, NoChange()), keys)
mgfcall_map(mgf::Diffed{<:MemoizedGenerativeFunction}, keys) = mgfcall_map(mgf, Diffed(keys, NoChange()))
function mgfcall_map(mgf::Diffed{MemoizedGenerativeFunction, NoChange}, keys::Diffed{<:Any, NoChange})
    Diffed(mgfcall_map(strip_diff(mgf), strip_diff(keys)), NoChange())
end

##################
# mgfcall_setmap #
##################
function mgfcall_setmap(mgf::MemoizedGenerativeFunction, keys)
    Set(mgf[key] for key in keys)
end

function mgfcall_setmap(mgf::Diffed{<:MemoizedGenerativeFunction}, keys::Diffed)
    diff = get_diff(mgf) === NoChange() && get_diff(keys) === NoChange() ? NoChange() : UnknownChange()
    Diffed(mgfcall_setmap(strip_diff(mgf), strip_diff(keys)), diff)
end