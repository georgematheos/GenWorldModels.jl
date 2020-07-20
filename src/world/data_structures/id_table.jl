struct IDTable{typenames}
    id_to_idx::NamedTuple{typenames, <:Tuple{Vararg{PersistentHashMap{UUID, Int}}}}
    idx_to_id::NamedTuple{typenames, <:Tuple{Vararg{PersistentHashMap{Int, UUID}}}}
end

function IDTable(types::Tuple)
    typenames = map(oupm_type_name, types)
    id_to_idx_maps = Tuple(PersistentHashMap{UUID, Int}() for _=1:length(types))
    idx_to_id_maps = Tuple(PersistentHashMap{Int, UUID}() for _=1:length(types))
    IDTable{()}(NamedTuple(), NamedTuple())
    IDTable{typenames}(NamedTuple{typenames}(id_to_idx_maps), NamedTuple{typenames}(idx_to_id_maps))
end

oupm_type_name(T) = Symbol(Base.typename(T))

@inline get_idx(table::IDTable, obj::T) where {T <: OUPMType} = get_idx(table, T, obj.idx_or_id)
@inline get_idx(table::IDTable, type::Type{<:OUPMType}, id::UUID) = get_idx(table, oupm_type_name(type), id)
@inline get_idx(table::IDTable, typename::Symbol, id::UUID) = table.id_to_idx[typename][id]

@inline has_id(table::IDTable, obj::T) where {T <: OUPMType} = has_id(table, T, obj.idx_or_id)
@inline has_id(table::IDTable, type::Type{<:OUPMType}, id::UUID) = has_id(table, oupm_type_name(type), id)
@inline has_id(table::IDTable, typename::Symbol, id::UUID) = haskey(table.id_to_idx[typename], id)

@inline has_idx(table::IDTable, type::Type{<:OUPMType}, idx::Int) = has_idx(table, oupm_type_name(type), idx)
@inline has_idx(table::IDTable, typename::Symbol, idx::Int) = haskey(table.idx_to_id[typename], idx)
@inline get_id(table::IDTable, type::Type{<:OUPMType}, idx::Int) = get_id(table, oupm_type_name(type), idx)
@inline get_id(table::IDTable, typename::Symbol, idx::Int) = table.idx_to_id[typename][idx]

@inline add_identifier_for(table::IDTable, type::Type{<:OUPMType}, idx::Int) = add_identifier_for(table, oupm_type_name(type), idx)
function add_identifier_for(table::IDTable, typename::Symbol, idx::Int)
    id = UUIDs.uuid4()
    new_table = insert_id(table, typename, idx, id)
    return (new_table, id)
end

"""
    insert_id(table, typename, idx, id)

Associate the `idx` with the given `id` and the `id` with the `idx`,
potentially overwriting the current associations for this `id` and `idx`.
"""
function insert_id(table::IDTable, typename::Symbol, idx::Int, id::UUID)
    new_id_to_idx_for_type = assoc(table.id_to_idx[typename], id, idx)
    new_idx_to_id_for_type = assoc(table.idx_to_id[typename], idx, id)
    new_id_to_idx = merge(table.id_to_idx, NamedTuple{(typename,)}((new_id_to_idx_for_type,)))
    new_idx_to_id = merge(table.idx_to_id, NamedTuple{(typename,)}((new_idx_to_id_for_type,)))
    IDTable(new_id_to_idx, new_idx_to_id)
end

"""
    move_all_between(table::IDTable, changed_ids::Set, typename::Symbol; min::Int, Inc::Int, max=Inf)

Change the id `table` so that all the ids currently associated
with indices between `min` and `max` (inclusive) have their associated indices
changed to be the current associated index plus `inc`.  (Ie. `new_idx = current_idx + inc`.)
Push every identifier whose index was changed into the set `changed_ids`.
Returns `new_table`.

This update may result in some indices which previously had associated identifiers
now having no associated identifier.
The update may also overwrite current associations for indices in the range
(min+inc,...min) or (max...max+inc).
"""
function move_all_between(table::IDTable, changed_ids::Set, typename::Symbol; min, inc, max=Inf)
    id_to_idx_for_type = table.id_to_idx[typename]
    idx_to_id_for_type = table.idx_to_id[typename]

    # TODO: don't scan the full table; only look at those which we actually need to change
    actual_max = 0
    for (idx, id) in table.idx_to_id[typename]
        if idx >= min && idx <= max
            actual_max = actual_max > idx ? actual_max : idx
            id_to_idx_for_type = assoc(id_to_idx_for_type, id, idx + inc)
            idx_to_id_for_type = assoc(idx_to_id_for_type, idx + inc, id)
            push!(changed_ids, id)
        end
    end
    max = actual_max
    # We have not changed the idx for each id for indices in the range,
    # and changed the id for indices in the range (min+inc, max+inc).
    # Now we need to change the ids for indices in range (min, min+inc-1) and (max+inc+1, max).
    # In particular, we change the ids by simply deleting the current association--others can be 
    # added later if need be.
    for idx=min:(min+inc-1)
        idx_to_id_for_type = dissoc(idx_to_id_for_type, idx)
    end
    for idx=(max+inc+1):max
        idx_to_id_for_type = dissoc(idx_to_id_for_type, idx)
    end

    new_id_to_idx = merge(table.id_to_idx, NamedTuple{(typename,)}((id_to_idx_for_type,)))
    new_idx_to_id = merge(table.idx_to_id, NamedTuple{(typename,)}((idx_to_id_for_type,)))
    new_table = IDTable(new_id_to_idx, new_idx_to_id)
    return new_table
end

# function move_identifier_from_index_to_index(table::IDTable, typename::Symbol; from_idx, to_idx)
#     id_to_idx_for_type = table.id_to_idx[typename]
#     idx_to_id_for_type = table.idx_to_id[typename]
#     id = idx_to_id_for_type[from_idx]
#     id_to_idx_for_type = assoc(id_to_idx_for_type, id, to_idx)
#     idx_to_id_for_type = assoc(idx_to_id_for_type, to_idx, id)
# end

@inline convert_key_to_id_form(key, id_table) = key
@inline function convert_key_to_id_form(key::OUPMType{Int}, id_table)
    type = oupm_type(key)
    id = get_id(id_table, type, key.idx_or_id)
    type(id)
end

@inline convert_key_to_idx_form(key, id_table) = key
@inline function convert_key_to_idx_form(key::OUPMType{UUID}, id_table)
    type = oupm_type(key)
    idx = get_idx(id_table, type, key.idx_or_id)
    type(idx)
end