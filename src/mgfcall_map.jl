###############
# mgfcall_map #
###############

struct MgfCallMapState{K, KI}
    keys::K
    keys_to_indices::KI
end
struct MgfCallMap <: Gen.CustomUpdateGF{Gen.LazyMap, MgfCallMapState} end
function Gen.apply_with_state(::MgfCallMap, (mgf, keys))
    (lazy_map(key -> mgf[key], keys), MgfCallMapState(keys, item_to_indices(keys)))
end
function Gen.update_with_state(::MgfCallMap, prev_state, (mgf, keys),
    (mgfdiff, keysdiff)::Tuple{WorldUpdateDiff, VectorDiff}, ::Selection
)
    new_keys_to_indices = prev_state.keys_to_indices
    for (i, diff) in keysdiff
        new_keys_to_indices = dissoc(new_keys_to_indices, prev_state.keys[i], i)
        new_keys_to_indices = assoc(new_keys_to_indices, keys[i], i)
    end

    if _current_update_came_from_queue(world(mgf)) && length(world(mgf).state.diffs) < length(keys)
        indices_to_diffs =  get_indices_changed_vars_via_diff_scan(Diffed(mgf, mgfdiff), new_keys_to_indices, keysdiff)
    else
        indices_to_diffs = get_indices_changed_vars_via_keys_scan(Diffed(mgf, mgfdiff), keys, keysdiff)
    end

    out_updated = Dict(
        indices_to_diffs..., (i => KeyChangedDiff(diff) for (i, diff) in keysdiff.updated)...
    )

    new_diff = VectorDiff(keysdiff.prev_length, keysdiff.new_length, out_updated)
    new_state = MgfCallMapState(keys, new_keys_to_indices)
    (new_state, LazyMap(key -> mgf[key], keys), new_diff)
end

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

##################
# mgfcall_setmap #
##################
struct MgfCallSetMap <: Gen.CustomUpdateGF{Gen.LazyBijectionSetMap, Nothing} end
function Gen.apply_with_state(::MgfCallSetMap, (mgf, keys))
    (lazy_bijection_set_map(key -> mgf[key], mgfcall -> key(mgfcall), keys), nothing)
end
function Gen.update_with_state(::MgfCallSetMap, _, (mgf, keys),
    (mgfdiff, keysdiff)::Tuple{WorldUpdateDiff, SetDiff}, ::Selection
)
    if _current_update_came_from_queue(world(mgf)) && length(world(mgf).state.diffs) < length(keys)
        diffed_keys =  get_set_changed_vars_via_diff_scan(Diffed(mgf, mgfdiff), keys)
    else
        diffed_keys = get_set_changed_vars_via_keys_scan(mgf, keys)
    end

    added = Set(
        (mgf[key] for key in Iterators.flatten((keysdiff.added, diffed_keys)))
    )
    deleted = Set(
        (mgf[key] for key in Iterators.flatten((keysdiff.deleted, diffed_keys)))
    )

    new_diff = SetDiff(added, deleted)
    (nothing, lazy_bijection_set_map(key -> mgf[key], mgfcall -> key(mgfcall), keys), new_diff)
end

const mgfcall_setmap = MgfCallSetMap()

function get_set_changed_vars_via_keys_scan(diffed_mgf, keys)
    Set(
        key for key in keys if get_diff(diffed_mgf[Diffed(key, NoChange())]) !== NoChange()
    )
end
function get_set_changed_vars_via_diff_scan(mgf, keys)
    Set(
        key(call) for call in keys(world(mgf).state.diffs) if addr(mgf) == addr(call) && key(call) in keys
    )
end

###################
# mgfcall_dictmap #
###################

struct MgfCallDictMapState{K, KI}
    key_to_mgfkey::K
    mgfkey_to_key::KI
end
struct MgfCallDictMap <: Gen.CustomUpdateGF{Gen.LazyValMapDict, MgfCallDictMapState} end
function Gen.apply_with_state(::MgfCallDictMap, (mgf, key_to_mgfkey))
    (lazy_val_map(mgfkey -> mgf[mgfkey], key_to_mgfkey), MgfCallDictMapState(key_to_mgfkey, vals_to_keys(key_to_mgfkey)))
end
function Gen.update_with_state(::MgfCallDictMap, prev_state, (mgf, key_to_mgfkey),
    (mgfdiff, key_to_mgfkey_diff)::Tuple{WorldUpdateDiff, DictDiff}, ::Selection
)
    new_mgfkey_to_key = prev_state.mgfkey_to_key
    for (key, diff) in keys_to_mgfkey_diff.updated
        new_mgfkey_to_key = dissoc(new_mgfkey_to_key, prev_state.key_to_mgfkey[key], key)
        new_mgfkey_to_key = assoc(new_mgfkey_to_key, key_to_mgfkey[key], key)
    end

    if _current_update_came_from_queue(world(mgf)) && length(world(mgf).state.diffs) < length(key_to_mgfkey)
        keys_to_diffs =  get_keys_changed_vars_via_diff_scan(Diffed(mgf, mgfdiff), new_mgfkey_to_key, keysdiff)
    else
        keys_to_diffs = get_keys_changed_vars_via_keys_scan(Diffed(mgf, mgfdiff), keys, keysdiff)
    end

    out_updated = Dict(
        keys_to_diffs..., (key => KeyChangedDiff(diff) for (key, diff) in key_to_mgfkey_diff.updated)...
    )

    new_diff = DictDiff(lazy_val_map(mgfkey -> mgf[mgfkey], key_to_mgfkey_diff.added), key_to_mgfkey_diff.deleted, out_updated)
    new_state = MgfCallDictMapState(key_to_mgfkey, new_mgfkey_to_key)
    (new_state, lazy_val_map(mgfkey -> mgfmgf[key], key_to_mgfkey), new_diff)
end

const mgfcall_dictmap = MgfCallDictMap()

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
    (mgfdiff, vecdiff)::Tuple{WorldUpdateDiff, NoChange}, s::Selection
)
    Gen.update_with_state(m, prev_state, (mgf, keys), (mgfdiff, VectorDiff(length(keys), length(keys), Dict())), s)
end
function Gen.update_with_state(m::MgfCallMap, prev_state, (mgf, keys),
    (mgfdiff, vecdiff)::Tuple{<:Diff, <:Diff}, s::Selection
)
    retval, state = Gen.apply_with_state(m, (mgf, keys))
    (state, retval, UnknownChange())
end

### mgfcall_setmap ###
function Gen.update_with_state(m::MgfCallSetMap, prev_state, (mgf, keys),
    (mgfdiff, vecdiff)::Tuple{WorldUpdateDiff, NoChange}, s::Selection
)
    Gen.update_with_state(m, prev_state, (mgf, keys), (mgfdiff, SetDiff(Set(), Set())), s)
end
function Gen.update_with_state(m::MgfCallSetMap, prev_state, (mgf, keys),
    (mgfdiff, vecdiff)::Tuple{<:Diff, <:Diff}, s::Selection
)
    retval, state = Gen.apply_with_state(m, (mgf, keys))
    (state, retval, UnknownChange())
end

### mgfcall_dictmap ###
function Gen.update_with_state(m::MgfCallDictMap, prev_state, (mgf, keys),
    (mgfdiff, vecdiff)::Tuple{WorldUpdateDiff, NoChange}, s::Selection
)
    Gen.update_with_state(m, prev_state, (mgf, keys), (mgfdiff, DictDiff(Dict(), Set(), Dict())), s)
end
function Gen.update_with_state(m::MgfCallDictMap, prev_state, (mgf, keys),
    (mgfdiff, vecdiff)::Tuple{<:Diff, <:Diff}, s::Selection
)
    retval, state = Gen.apply_with_state(m, (mgf, keys))
    (state, retval, UnknownChange())
end