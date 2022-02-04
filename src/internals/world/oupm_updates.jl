###############
# Conversions #
###############

function get_with_abstract_origin(world, concrete::ConcreteIndexOUPMObject{T}) where {T}
    c_to_a(x) = convert_to_abstract(world, x)
    new_origin = map(c_to_a, concrete.origin)
    ConcreteIndexOUPMObject{T}(new_origin, concrete.idx)
end

function get_or_generate_with_abstract_origin!(world, concrete::ConcreteIndexOUPMObject{T}) where {T}
    c_to_a(x) = convert_to_abstract!(world, x)
    new_origin = map(c_to_a, concrete.origin)
    ConcreteIndexOUPMObject{T}(new_origin, concrete.idx)
end

function try_to_get_with_concrete_origin(id_table, concrete::ConcreteIndexAbstractOriginOUPMObject{T}) where {T}
    function a_to_c(a)
        if haskey(id_table, a)
            try_to_get_with_concrete_origin(id_table, id_table[a])
        else
            a
        end
    end
    new_origin = map(a_to_c, concrete.origin)
    ConcreteIndexOUPMObject{T}(new_origin, concrete.idx)
end
try_to_get_with_concrete_origin(_, c::ConcreteIndexOUPMObject) = c

#########################
# Top-level bookkeeping #
#########################

"""
    perform_oupm_moves_and_enqueue_downstream!(world, oupm_updates)

Performs all the oupm updates in `oupm_updates` in order and enqueues
any calls which look at the indices for changed identifiers.

This will also ensure add information to `world.state.origins_with_id_table_updates`
about every origin which has had a change in the ID table for it.
"""
function perform_oupm_moves_and_enqueue_downstream!(world::World, oupm_updates)
    # I'm currently "betting" that the overhead of using sets rather than lists outweighs
    # the risk of iterating over an item twice.  I could change these lists to sets, though.  Or linked lists?
    origin_changed = AbstractOUPMObject[]
    index_changed = AbstractOUPMObject[]
    abstract_changed = ConcreteIndexAbstractOriginOUPMObject[]
    original_id_table = world.id_table
    for oupm_update in oupm_updates
        # this will perform the move in the world, and `push!` all the objects with origin, index, and abstract changed to the lists
        updated_typenames_and_origins = perform_oupm_move!(world, oupm_update, origin_changed, index_changed, abstract_changed)
        note_new_origin_changes!(world.state.origins_with_id_table_updates, updated_typenames_and_origins)
    end
    enqueue_downstream_from_oupm_moves!(world, original_id_table, origin_changed, index_changed, abstract_changed)
end

"""
    note_new_origin_changes!(origins_with_id_table_updates, updated_typenames_and_origins)

Given an iterator `updated_typenames_and_origins` of pairs `(typename, set_of_origins_for_this_type_with_changes_in_id_table)`,
this will note in the `origins_with_id_table_updates` that these origins have had changes for this type (in addition
to whatever changes for this type are already noted).

`origins_with_id_table_updates` should be a table with entries `typename => set_of_origins_for_this_type_with_changes_in_id_table`
we are going to add to.
"""
function note_new_origin_changes!(origins_with_id_table_updates, updated_typenames_and_origins)
    for (typename, origins) in updated_typenames_and_origins
        if !haskey(origins_with_id_table_updates, typename)
            origins_with_id_table_updates[typename] = origins
        else
            union!(origins_with_id_table_updates[typename], origins)
        end
    end
end

"""
    enqueue_downstream_from_oupm_moves!(world::World, original_id_table, origin_changed, index_changed, abstract_changed)

Enqueue all calls which look up conversions of the OUPM objects changed by the OUPM updates which left remnants
`origin_changed`, `index_changed`, and `abstract_changed`.
"""
function enqueue_downstream_from_oupm_moves!(world::World, original_id_table, origin_changed, index_changed, abstract_changed)
    for abstract in index_changed
        c1 = Call(_get_index_addr, abstract)
        c2 = Call(_get_concrete_addr, abstract)
        world.state.diffs[c1] = UnknownChange()
        world.state.diffs[c2] = UnknownChange()
        enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, c1)
        enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, c2)
    end
    for abstract in origin_changed
        c1 = Call(_get_origin_addr, abstract)
        c2 = Call(_get_concrete_addr, abstract)
        world.state.diffs[c1] = UnknownChange()
        world.state.diffs[c2] = UnknownChange()
        enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, c1)
        enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, c2)
    end
    for concrete in abstract_changed
        c = Call(_get_abstract_addr, concrete)
        if has_val(world, c)
            # If this call to `abstract` did not exist before,
            # but it exists now [probably because an abstract object was shifted up
            # in the indexing scheme by a creation below it],
            # generate an entry in the dependency tracker for it here
            # to maintain the invariant that if `has_val(world, c)`, then
            # there is a tracker entry for `c` in the world.
            note_new_call_if_none_exists!(world, c)
        end
        if haskey(original_id_table, concrete)
            # If there were an `abstract` for this object before,
            # it is possible that calls looked it up; enqueue these calls for processing
            world.state.diffs[c] = UnknownChange()
            enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, c)
        end
    end
end

function move_all_between!(world::World, typename::Symbol, origin::Tuple{Vararg{<:AbstractOUPMObject}}, index_changed, abstract_changed; min=1, inc, max=Inf)
    new_id_table = move_all_between(world.id_table, typename, origin, index_changed, abstract_changed; min=min, max=max, inc=inc)
    world.id_table = new_id_table
    world
end

####################
# Performing moves #
####################

"""
    perform_oupm_move!(world::World, spec::OUPMMove, origin_changed, index_changed, abstract_changed)

Performs the OUPM move specified by `spec` in the world.  Pushes all abstract objects
whose origins have changed to `origin_changed`, all abstract objects whose index have changed
to `index_changed`, and all concrete objects whose abstract forms have changed to `abstract_changed`.

Returns an iterator over pairs `(typename, origins_for_type_with_changes)`,
where `origins_for_type_with_changes` is a set containing all origins for this typename
for which the ID associations have been updated.
"""
function perform_oupm_move!(world::World, spec::Move, origin_changed, index_changed, abstract_changed)
    from = get_with_abstract_origin(world, spec.from)
    to = get_or_generate_with_abstract_origin!(world, spec.to)
    
    has_abstract = haskey(world.id_table, from)
    if has_abstract
        abstract = world.id_table[from]
    end
    if from.origin == to.origin
        if from.idx < to.idx
            min = from.idx + 1
            max = to.idx
            inc = -1
        else
            min = to.idx
            max = from.idx - 1
            inc = +1
        end
        move_all_between!(world, oupm_type_name(from), from.origin, index_changed, abstract_changed; max=max, min=min, inc=inc)
    else
        # move down all above `from`
        move_all_between!(world, oupm_type_name(from), from.origin, index_changed, abstract_changed; min=from.idx+1, max=Inf, inc=-1)
        # move up all after and including `to`
        move_all_between!(world, oupm_type_name(to), to.origin, index_changed, abstract_changed; min=to.idx, max=Inf, inc=+1)

        if has_abstract
            push!(origin_changed, abstract)
        end
    end
    if has_abstract
        push!(index_changed, abstract)
        push!(abstract_changed, to)
        world.id_table = assoc(world.id_table, abstract, to)
    end
    
    return (oupm_type_name(from) => Set([from.origin, to.origin]),)
end

function perform_oupm_move!(world::World, spec::Create{T}, _, index_changed, abstract_changed) where {T}
    obj = get_or_generate_with_abstract_origin!(world, spec.obj)
    move_all_between!(world, T, obj.origin, index_changed, abstract_changed; min=obj.idx, inc=+1)
    return (oupm_type_name(obj) => Set(obj.origin),)
end

function perform_oupm_move!(world::World, spec::Delete{T}, _, index_changed, abstract_changed) where {T}
    obj = get_or_generate_with_abstract_origin!(world, spec.obj)
    move_all_between!(world, T, obj.origin, index_changed, abstract_changed; min=obj.idx+1, inc=-1)
    return (oupm_type_name(obj) => Set(obj.origin),)
end

function perform_oupm_move!(world, spec::Split{T}, origin_changed, index_changed, abstract_changed) where {T}
    from = get_with_abstract_origin(world, spec.from)
    from_idx = from.idx
    to_idx1 = min(spec.to_idx_1, spec.to_idx_2)
    to_idx2 = max(spec.to_idx_1, spec.to_idx_2)

    changed_origins = Dict()
    changed_origins[oupm_type_name(from)] = Set(from.origin)

    # before we change the abstract associations, get the current abstract for for the objects we are
    # going to need to move
    moves = map(
        ((_from, _to),) -> ((get_with_abstract_origin(world, _from)) => _to),
        spec.moves
    )
    if moves != ()
        abstract_from = world.id_table[from]
    end

    # capture the variables common to all the `move_all_between!` calls we need, to simplify syntax
    mb!(min, max, inc) = move_all_between!(world, T, from.origin, index_changed, abstract_changed; min=min, max=max, inc=inc)

    if from_idx < to_idx1
        mb!(from_idx + 1, to_idx1, -1),
        mb!(to_idx2, Inf, +1)
    elseif from_idx < to_idx2
        mb!(to_idx1, from_idx - 1, +1),
        mb!(to_idx2, Inf, +1)
    else
        mb!(from_idx + 1, Inf, +1),
        mb!(to_idx2 - 1, from_idx - 1, +2),
        mb!(to_idx1, to_idx2 - 2, +1)
    end

    # now that the new associations are present, we can move the objects for which moves have been specified
    if length(moves) > 0
        perform_moves_after_splitmerge!(world, moves, (abstract_from,), abstract_changed, index_changed, origin_changed, changed_origins)
    end

    return changed_origins
end

function perform_oupm_move!(world, spec::Merge{T}, origin_changed, index_changed, abstract_changed) where {T}
    to = get_or_generate_with_abstract_origin!(world, spec.to)
    to_idx = to.idx
    from_idx1 = min(spec.from_idx_1, spec.from_idx_2)
    from_idx2 = max(spec.from_idx_1, spec.from_idx_2)

    changed_origins = Dict()
    changed_origins[oupm_type_name(to)] = Set(to.origin)

    # capture the variables common to all the `move_all_between!` calls we need, to simplify syntax
    mb!(min, max, inc) = move_all_between!(world, T, to.origin, index_changed, abstract_changed; min=min, max=max, inc=inc)

    moves = map(
        ((_from, _to),) -> (get_with_abstract_origin(world, _from) => _to),
        spec.moves
    )
    if moves != ()
        abstract_from_1 = world.id_table[ConcreteIndexOUPMObject{T}(to.origin, spec.from_idx_1)]
        abstract_from_2 = world.id_table[ConcreteIndexOUPMObject{T}(to.origin, spec.from_idx_2)]
    end

    if to_idx < from_idx1
        mb!(from_idx2 + 1, Inf, -1),
        mb!(to_idx, from_idx1 - 1, +1)
    elseif to_idx < from_idx2
        mb!(from_idx2 + 1, Inf, -1),
        mb!(from_idx1 + 1, to_idx, -1)
    else
        mb!(from_idx1 + 1, from_idx2 - 1, -1),
        mb!(from_idx2 + 1, to_idx + 1, -2),
        mb!(to_idx + 2, Inf, -1)
    end

    if length(moves) > 0
        perform_moves_after_splitmerge!(world, moves, (abstract_from_1, abstract_from_2), abstract_changed, index_changed, origin_changed, changed_origins)
    end

    return changed_origins
end

"""
    note_origin_change_for_object!(changed_origins, obj)

Note in `changed_origins::Dict{TypeName, SetOfOriginsForType}` that
the given `obj` has had a change--so it's origin has experienced a change for this typename.
"""
function note_origin_change_for_object!(changed_origins, obj)
    typename = oupm_type_name(obj)
    if haskey(changed_origins, typename)
        push!(changed_origins[typename], obj.origin)
    else
        changed_origins[typename] = Set((obj.origin,))
    end
end

function perform_moves_after_splitmerge!(world, moves, abstract_from_objects, abstract_changed, index_changed, origin_changed, changed_origins)
    for (_from, _to) in moves
        note_origin_change_for_object!(changed_origins, _from)

        if _to === nothing
             world.id_table = dissoc(world.id_table, _from)
            push!(abstract_changed, _from)
            continue
        end

        @assert any(obj in _from.origin for obj in abstract_from_objects) "A move within $spec was specified to move an object $obj which does not have an object being split or merged in its origin!"
        _to = get_or_generate_with_abstract_origin!(world, _to)
        @assert _from.origin != _to.origin "A move within $spec attempted to perform a move which does not change an object's origin."

        note_origin_change_for_object!(changed_origins, _to)

        abst = world.id_table[_from]
        world.id_table = dissoc(world.id_table, _from)
        world.id_table = assoc(world.id_table, abst, _to)
        push!(origin_changed, abst)
        if _to.idx != _from.idx
            push!(index_changed, abst)
        end
        push!(abstract_changed, _from)
        push!(abstract_changed, _to)
    end
end