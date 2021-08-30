"""
    key_allows_diff_scan(mgf_addr, key)

Returns whether the given `key` is of a type which means diff scanning will yield
a valid update for a MGFCallMap with the given MGF address.
"""
function key_allows_diff_scan(mgf_addr, key)
    !(
        key isa Tuple{Vararg{<:ConcreteIndexOUPMObject}}
        || (key isa ConcreteIndexOUPMObject && !(key isa ConcreteIndexAbstractOriginOUPMObject))
        || (key isa ConcreteIndexAbstractOriginOUPMObject && mgf_addr != _get_abstract_addr)
    )
end

###############
# mgfcall_map #
###############

struct MgfCallMapState{K, KI}
    keys::K
    keys_to_indices::KI
    # this will be false if there are any MGFKeys which are tuples of concrete objects,
    # or keys which are concrete objects--if the mgf address is not `:abstract`
    mgfkeytypes_valid_for_diff_scan::Bool
end
struct MgfCallMap <: Gen.CustomUpdateGF{Gen.LazyMap, MgfCallMapState} end
function Gen.apply_with_state(::MgfCallMap, (mgf, keys))
    mgfkeytypes_valid_for_diff_scan = all(
        key_allows_diff_scan(addr(mgf), mgfkey) for mgfkey in keys
    )
    (lazy_map(key -> mgf[key], keys), MgfCallMapState(keys, item_to_indices(keys), mgfkeytypes_valid_for_diff_scan))
end
function Gen.update_with_state(::MgfCallMap, prev_state, (mgf, keys),
    (mgfdiff, keysdiff)::Tuple{WorldUpdateDiff, VectorDiff}
)

    new_keys_to_indices = prev_state.keys_to_indices
    for (i, diff) in keysdiff.updated
        new_keys_to_indices = dissoc(new_keys_to_indices, prev_state.keys[i], i)
        new_keys_to_indices = assoc(new_keys_to_indices, keys[i], i)
    end

    indices_to_diffs = get_indices_to_diff_vect(mgf, mgfdiff, keysdiff, prev_state)

    out_updated = Dict(
        indices_to_diffs..., (i => KeyChangedDiff(diff) for (i, diff) in keysdiff.updated if diff != NoChange())...
    )

    mgfkeytypes_valid_for_diff_scan = prev_state.mgfkeytypes_valid_for_diff_scan && all(
        key_allows_diff_scan(addr(mgf), mgfkey)
        for mgfkey in Iterators.flatten((
            (keys[i] for i=max(1, keysdiff.prev_length):keysdiff.new_length),
            (keys[i] for (i, _) in keysdiff.updated),
        ))
    )

    new_diff = VectorDiff(keysdiff.prev_length, keysdiff.new_length, out_updated)
    new_state = MgfCallMapState(keys, new_keys_to_indices, mgfkeytypes_valid_for_diff_scan)

    (new_state, lazy_map(key -> mgf[key], keys), new_diff)
end

function get_indices_to_diff_vect(mgf, mgfdiff, keysdiff, prev_state)
    if !(prev_state.mgfkeytypes_valid_for_diff_scan)
        @warn """
        A lookup_or_generate_map is being applied to a vector which contains values which are either
        - A tuple of `ConcreteIndexOUPMObject`s
        - A `ConcreteIndexAbstractOriginOUPMObject` being used in a situation other than looking up `:abstract`
        - A `ConcreteIndexOriginOUPMObject` which is not a `ConcreteIndexAbstractOriginOUPMObject`
        As a result, some performance optimizations cannot be applied.
        """
    end

    can_do_diff_scan = prev_state.mgfkeytypes_valid_for_diff_scan && _current_update_came_from_queue(world(mgf))

    if can_do_diff_scan && length(world(mgf).state.diffs) < length(prev_state.keys)
        indices_to_diffs =  get_indices_changed_vars_via_diff_scan(Diffed(mgf, mgfdiff), prev_state.keys_to_indices, keysdiff)
    else
        indices_to_diffs = get_indices_changed_vars_via_keys_scan(Diffed(mgf, mgfdiff), prev_state.keys, keysdiff)
    end    
end

"""
    mgf_calls ~ mgfcall_map(mgf, keys::AbstractVector)

Returns `[mgf[key] for key in keys]`, where `mgf` is a memoized generative function;
propagates diffs so that some `mgf[key]`s whose values in the world have not change
will have a `NoChange` diff (and attempts to do so efficiently.)
"""
const mgfcall_map = MgfCallMap()

# these functions return a mapping idx -> mgfcalldiff, for each idx which
# may have a change (if it is valid to call this function), if that idx
# is not in keys(keysdiff.updated)
function get_indices_changed_vars_via_keys_scan(diffed_mgf, keys, keysdiff)
    idx_to_diff = Dict()
    for (i, key) in enumerate(keys)
        if haskey(keysdiff.updated, i)
            continue
        end
        mgfcalldiff = get_diff(diffed_mgf[Diffed(keys[i], NoChange())])
        if mgfcalldiff !== NoChange()
            idx_to_diff[i] = mgfcalldiff
        end
    end
    idx_to_diff
end
function get_indices_changed_vars_via_diff_scan(diffed_mgf, keys_to_indices, keysdiff)
    idx_to_diff = Dict()
    mgf = strip_diff(diffed_mgf)
    for (call, diff) in world(mgf).state.diffs
        # if addr(call) == addr(mgf) || addr(call) = _get_abstract_addr
        #     # we need to consider all lookups for this specific key, and any lookup which
        #     # may be converting this key to abstract, to then have its value looked up
        #     if haskey(keys_to_indices, key(call))

        #     end

        if addr(mgf) == addr(call) && haskey(keys_to_indices, key(call))
            no_keychange_diff = get_diff(diffed_mgf[Diffed(key(call), NoChange())])
            for i in keys_to_indices[key(call)]
                if !haskey(keysdiff.updated, i)
                    idx_to_diff[i] = no_keychange_diff
                end
            end
        end
    end
    idx_to_diff
end

###################
# mgfcall_dictmap #
###################

struct MgfCallDictMapState{K, KI}
    key_to_mgfkey::K
    mgfkey_to_key::KI
    # this will be false if there are any MGFKeys which are tuples of concrete objects,
    # or keys which are concrete objects--if the mgf address is not `:abstract`
    mgfkeytypes_valid_for_diff_scan::Bool
end
struct MgfCallDictMap <: Gen.CustomUpdateGF{Gen.LazyValMapDict, MgfCallDictMapState} end
function Gen.apply_with_state(::MgfCallDictMap, (mgf, key_to_mgfkey))
    mgfkeytypes_valid_for_diff_scan = all(
        key_allows_diff_scan(addr(mgf), mgfkey) for (_, mgfkey) in key_to_mgfkey
    )
    (lazy_val_map(mgfkey -> mgf[mgfkey], key_to_mgfkey), MgfCallDictMapState(key_to_mgfkey, vals_to_keys(key_to_mgfkey), mgfkeytypes_valid_for_diff_scan))
end
function Gen.update_with_state(::MgfCallDictMap, prev_state, (mgf, key_to_mgfkey)::Tuple{Any, AbstractDict{<:Any, MGFKeyType}},
    (mgfdiff, key_to_mgfkey_diff)::Tuple{WorldUpdateDiff, DictDiff}
) where {MGFKeyType}
    new_mgfkey_to_key = prev_state.mgfkey_to_key
    for (key, diff) in key_to_mgfkey_diff.updated
        new_mgfkey_to_key = dissoc(new_mgfkey_to_key, prev_state.key_to_mgfkey[key], key)
        new_mgfkey_to_key = assoc(new_mgfkey_to_key, key_to_mgfkey[key], key)
    end

    keys_to_diffs = get_keys_to_diff_dict(mgf, mgfdiff, key_to_mgfkey_diff, prev_state)

    out_updated = Dict{Any, Diff}(
        keys_to_diffs..., (key => KeyChangedDiff(diff) for (key, diff) in key_to_mgfkey_diff.updated)...
    )

    mgfkeytypes_valid_for_diff_scan = prev_state.mgfkeytypes_valid_for_diff_scan && all(
        key_allows_diff_scan(addr(mgf), mgfkey)
        for mgfkey in Iterators.flatten((
            (mgfkey for (_, mgfkey) in key_to_mgfkey_diff.added),
            (key_to_mgfkey[k] for (k, _) in key_to_mgfkey_diff.updated),
        ))
    )

    new_diff = DictDiff(lazy_val_map(mgfkey -> mgf[mgfkey], key_to_mgfkey_diff.added), key_to_mgfkey_diff.deleted, out_updated)
    new_state = MgfCallDictMapState(key_to_mgfkey, new_mgfkey_to_key, mgfkeytypes_valid_for_diff_scan)
    (new_state, lazy_val_map(mgfkey -> mgf[mgfkey], key_to_mgfkey), new_diff)
end

"""
    key_to_mgf_call ~ mgfcall_dictmap(mgf, key_to_mgfcallkey::AbstractDict)

Returns `Dict(k => mgf[v] for (k, v) in key_to_mgfcallkey]`, where `mgf` is a memoized generative function;
propagates diffs so that some `mgf[v]`s whose values in the world have not change
will have a `NoChange` diff (and attempts to do so efficiently.)
"""
const mgfcall_dictmap = MgfCallDictMap()

function get_keys_to_diff_dict(mgf, mgfdiff, key_to_mgfkey_diff, prev_state)
    key_to_mgfkey = prev_state.key_to_mgfkey; mgfkey_to_key = prev_state.mgfkey_to_key

    if !(prev_state.mgfkeytypes_valid_for_diff_scan)
        @warn """
        A lookup_or_generate_dictmap is being applied to a dictionary which contains values which are either
        - A tuple of `ConcreteIndexOUPMObject`s
        - A `ConcreteIndexAbstractOriginOUPMObject` being used in a situation other than looking up `:abstract`
        - A `ConcreteIndexOriginOUPMObject` which is not a `ConcreteIndexAbstractOriginOUPMObject`
        As a result, some performance optimizations cannot be applied.
        """
    end

    can_do_diff_scan = prev_state.mgfkeytypes_valid_for_diff_scan && _current_update_came_from_queue(world(mgf))

    if can_do_diff_scan && length(world(mgf).state.diffs) < length(key_to_mgfkey)
        keys_to_diffs =  get_keys_changed_vars_via_diff_scan(Diffed(mgf, mgfdiff), mgfkey_to_key, key_to_mgfkey_diff)
    else
        keys_to_diffs = get_keys_changed_vars_via_keys_scan(Diffed(mgf, mgfdiff), key_to_mgfkey, key_to_mgfkey_diff)
    end
end

function get_keys_changed_vars_via_keys_scan(diffed_mgf, key_to_mgfkey, keysdiff)
    key_to_diff = Dict()
    for (key, mgfkey) in key_to_mgfkey
        if haskey(keysdiff.updated, key)
            continue
        end
        mgfcalldiff = get_diff(diffed_mgf[Diffed(mgfkey, NoChange())])
        if mgfcalldiff !== NoChange()
            key_to_diff[key] = mgfcalldiff
        end
    end
    key_to_diff
end

# invariant: if we call this, MGF keys cannot be tuples of concrete index objects
# and if they are concrete, this must be an `abstract` lookup
function get_keys_changed_vars_via_diff_scan(diffed_mgf, mgfkey_to_key, keysdiff)
    key_to_diff = Dict()
    mgf = strip_diff(diffed_mgf)

    for (call, diff) in world(mgf).state.diffs
        if addr(mgf) == addr(call) && haskey(mgfkey_to_key, key(call))
            no_keychange_diff = get_diff(diffed_mgf[Diffed(key(call), NoChange())])
            for key in mgfkey_to_key[key(call)]
                if !haskey(keysdiff.updated, key) && !(key in keysdiff.deleted)
                    key_to_diff[key] = no_keychange_diff
                end
            end
        end
    end
    key_to_diff
end

####################################################
# diff propagation for `NoChange`, `UnknownChange` #
####################################################

### mgfcall_map ###
function Gen.update_with_state(m::MgfCallMap, prev_state, (mgf, keys),
    (mgfdiff, vecdiff)::Tuple{WorldUpdateDiff, NoChange}
)
    Gen.update_with_state(m, prev_state, (mgf, keys), (mgfdiff, VectorDiff(length(keys), length(keys), Dict())))
end
function Gen.update_with_state(m::MgfCallMap, prev_state, (mgf, keys),
    (mgfdiff, vecdiff)::Tuple{<:Diff, <:Diff}
)
    retval, state = Gen.apply_with_state(m, (mgf, keys))
    (state, retval, UnknownChange())
end

### mgfcall_dictmap ###
function Gen.update_with_state(m::MgfCallDictMap, prev_state, (mgf, keys),
    (mgfdiff, vecdiff)::Tuple{WorldUpdateDiff, NoChange}
)
    Gen.update_with_state(m, prev_state, (mgf, keys), (mgfdiff, DictDiff(Dict(), Set(), Dict{Any, Diff}())))
end
function Gen.update_with_state(m::MgfCallDictMap, prev_state, (mgf, keys),
    (mgfdiff, vecdiff)::Tuple{<:Diff, <:Diff}
)
    retval, state = Gen.apply_with_state(m, (mgf, keys))
    (state, retval, UnknownChange())
end