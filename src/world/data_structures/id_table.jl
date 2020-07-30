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

function Base.show(io::IO, table::IDTable{typenames}) where {typenames}
    # we should always have the same ids and idxs in the idx_to_id and id_to_idx table,
    # but we retain the capacity to print buggy, nonsymmetric tables to help with debugging
    println("----------------- ID Table -----------------")
    for name in typenames
        println(io, "$name")
        for (idx, id) in table.idx_to_id[name]
            if haskey(table.id_to_idx[name], id) && table.id_to_idx[name][id] == idx
                println(io, " $idx <-> $id")
            end
        end
        for (idx, id) in table.idx_to_id[name]
            if !haskey(table.id_to_idx[name], id) || table.id_to_idx[name][id] !== idx
                println(io, " $idx --> $id")
            end
        end
        for (id, idx) in table.id_to_idx[name]
            if !haskey(table.idx_to_id[name], idx) || table.idx_to_id[name][idx] !== id
                println(io, " $idx <-- $id")
            end
        end
    end
    println("--------------------------------------------")
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

# ALL instances of generating a new ID should pass through this function
@inline add_identifier_for(table::IDTable, type::Type{<:OUPMType}, idx::Int) = add_identifier_for(table, oupm_type_name(type), idx)
function add_identifier_for(table::IDTable, typename::Symbol, idx::Int)
    id = UUIDs.uuid1()
    new_table = insert_id(table, typename, idx, id)
    return (new_table, id)
end

"""
    insert_id(table, typename, idx, id)

Associate the `idx` with the given `id` and the `id` with the `idx`,
potentially overwriting the current associations for this `id` and `idx`.
"""
function insert_id(table::IDTable, typename::Symbol, idx::Int, id::UUID)
    @assert !haskey(table.id_to_idx[typename], id) "we should never be inserting an id which already exists!"
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
function move_all_between(table::IDTable, typename::Symbol; min=1, inc, max=Inf)
    id_to_idx_for_type = table.id_to_idx[typename]
    idx_to_id_for_type = table.idx_to_id[typename]
    original_idx_to_id = idx_to_id_for_type
    
    changed_ids = Set{UUID}()
    for (idx, id) in original_idx_to_id
        # at each step through this loop, we will update
        # id_to_idx_for_type[id].
        # if we are moving this ID, we will also update idx_to_id_for_type[idx+inc].
        # if there is no ID which will be moved to position `idx`, we also
        # update idx_to_id_for_type[idx]
        if (min <= idx && idx <= max)
            # all ids in the min<=...<=max range have their index incremented by `inc`
            id_to_idx_for_type = assoc(id_to_idx_for_type, id, idx+inc)
            idx_to_id_for_type = assoc(idx_to_id_for_type, idx+inc, id)
            push!(changed_ids, id)
        elseif (min+inc <= idx && idx < min) || (max < idx && idx <= max+inc)
            # all ids with current index in an area IDs are getting moved to, but where
            # ids are not moving anywhere, should be removed since they are going to be overwritten
            id_to_idx_for_type = dissoc(id_to_idx_for_type, id)
            push!(changed_ids, id)
        end

        if (min+inc <= idx && idx <= max+inc)
            # all indices in the range where IDs are being moved will either
            # 1. have an ID moved to this index, in which case the above `if` statement handles the id change
            # or
            # 2. have no ID which is going to get moved to it, and hence needs to have its current id deleted
            # we handle (2) here.
            if !haskey(original_idx_to_id, idx-inc)
                idx_to_id_for_type = dissoc(idx_to_id_for_type, idx)
            end
        elseif (min <= idx && idx < min+inc) || (max+inc < idx && idx <= max)
            # all indices in an area where IDs are moving from, but no IDs are moving to,
            # should be removed
            idx_to_id_for_type = dissoc(idx_to_id_for_type, idx)
        end
    end

    new_id_to_idx = merge(table.id_to_idx, NamedTuple{(typename,)}((id_to_idx_for_type,)))
    new_idx_to_id = merge(table.idx_to_id, NamedTuple{(typename,)}((idx_to_id_for_type,)))
    new_table = IDTable(new_id_to_idx, new_idx_to_id)
    return (new_table, changed_ids)
end