struct IDTable{typenames}
    id_to_idx::NamedTuple{typenames, <:Tuple{Vararg{PersistentHashMap{UUID, Int}}}}
    idx_to_id::NamedTuple{typenames, <:Tuple{Vararg{PersistentHashMap{Int, UUID}}}}
end

function IDTable(types::Tuple)
    typenames = map(type -> Symbol(type), types)
    id_to_idx_maps = Tuple(PersistentHashMap{UUID, Int}() for _=1:length(types))
    idx_to_id_maps = Tuple(PersistentHashMap{Int, UUID}() for _=1:length(types))
    IDTable{()}(NamedTuple(), NamedTuple())
    IDTable{typenames}(NamedTuple{typenames}(id_to_idx_maps), NamedTuple{typenames}(idx_to_id_maps))
end

@inline get_idx(table::IDTable, obj::T) where {T <: OUPMType} = get_idx(table, T, obj.id)
@inline get_idx(table::IDTable, type::Type{<:OUPMType}, id::UUID) = get_idx(table, Symbol(type), id)
@inline get_idx(table::IDTable, typename::Symbol, id::UUID) = table.id_to_idx[typename][id]

@inline has_id(table::IDTable, obj::T) where {T <: OUPMType} = has_id(table, T, obj.id)
@inline has_id(table::IDTable, type::Type{<:OUPMType}, id::UUID) = has_id(table, Symbol(type), id)
@inline has_id(table::IDTable, typename::Symbol, id::UUID) = haskey(table.id_to_idx[typename], id)

@inline function _lookup_or_generate_identifier(table::IDTable, type::Type{<:OUPMType}, idx::Int)
    _lookup_or_generate_identifier(table, Symbol(type), idx)
end
function _lookup_or_generate_identifier(table::IDTable, typename::Symbol, idx::Int)
    if haskey(table.idx_to_id[typename], idx)
        (table.idx_to_id[typename][idx], table, false)
    else
        id = UUIDs.uuid4()
        new_table = insert_id(table, typename, idx, id)
        (id, new_table, true)
    end
end

function insert_id(table::IDTable, typename::Symbol, idx::Int, id::UUID)
    new_id_to_idx_for_type = assoc(table.id_to_idx[typename], id, idx)
    new_idx_to_id_for_type = assoc(table.idx_to_id[typename], idx, id)
    new_id_to_idx = merge(table.id_to_idx, NamedTuple{(typename,)}((new_id_to_idx_for_type,)))
    new_idx_to_id = merge(table.idx_to_id, NamedTuple{(typename,)}((new_idx_to_id_for_type,)))
    IDTable(new_id_to_idx, new_idx_to_id)
end

function move_all_between(table::IDTable, typename::Symbol; min, inc, max=Inf)
    id_to_idx_for_type = table.id_to_idx[typename]
    idx_to_id_for_type = table.idx_to_id[typename]
    changed_ids = UUID[]

    # TODO: don't scan the full table; only look at those which we actually need to change
    for (idx, id) in table.idx_to_id[typename]
        if idx >= min && idx <= max
            id_to_idx_for_type = assoc(id_to_idx_for_type, id, idx + inc)
            idx_to_id_for_type = assoc(idx_to_id_for_type, idx + inc, id)
            push!(changed_ids, id)
        end
    end
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
    return (new_table, changed_ids)
end