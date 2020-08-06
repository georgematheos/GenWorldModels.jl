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

"""
    perform_oupm_moves_and_enqueue_downstream!(world, oupm_updates)

Performs all the oupm updates in `oupm_updates` in order, and enqueues
any calls which look at the indices for changed identifiers.
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
        perform_oupm_move!(world, oupm_update, origin_changed, index_changed, abstract_changed)
    end
    enqueue_downstream_from_oupm_moves!(world, original_id_table, origin_changed, index_changed, abstract_changed)
end

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
        if !haskey(original_id_table, concrete)
            note_new_call!(world, c)
        end
        world.state.diffs[c] = UnknownChange()
        enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, c)
    end
end

function move_all_between!(world::World, typename::Symbol, origin::Tuple{Vararg{<:AbstractOUPMObject}}, index_changed, abstract_changed; min=1, inc, max=Inf)
    new_id_table = move_all_between(world.id_table, typename, origin, index_changed, abstract_changed; min=min, max=max, inc=inc)
    world.id_table = new_id_table
    world
end

function perform_oupm_move!(world::World, spec::MoveMove, origin_changed, index_changed, abstract_changed)
    from = get_with_abstract_origin(world, spec.from)
    to = get_or_generate_with_abstract_origin!(world, spec.to)
    
    has_abstract = haskey(world.id_table, from)
    if has_abstract
        abstract = world.id_table[from]
    end
    @assert from != to
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
end

function perform_oupm_move!(world::World, spec::BirthMove{T}, _, index_changed, abstract_changed) where {T}
    obj = get_or_generate_with_abstract_origin!(world, spec.obj)
    move_all_between!(world, T, obj.origin, index_changed, abstract_changed; min=obj.idx, inc=+1)
end

function perform_oupm_move!(world::World, spec::DeathMove{T}, _, index_changed, abstract_changed) where {T}
    obj = get_or_generate_with_abstract_origin!(world, spec.obj)
    move_all_between!(world, T, obj.origin, index_changed, abstract_changed; min=obj.idx+1, inc=-1)
end

function perform_oupm_move!(world, spec::SplitMove{T}, origin_changed, index_changed, abstract_changed) where {T}
    from = get_with_abstract_origin(world, spec.from)
    from_idx = from.idx
    to_idx1 = min(spec.to_idx_1, spec.to_idx_2)
    to_idx2 = max(spec.to_idx_1, spec.to_idx_2)

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
        perform_moves_after_splitmerge!(world, moves, (abstract_from,), abstract_changed, index_changed, origin_changed)
    end
end

function perform_oupm_move!(world, spec::MergeMove{T}, origin_changed, index_changed, abstract_changed) where {T}
    to = get_or_generate_with_abstract_origin!(world, spec.to)
    to_idx = to.idx
    from_idx1 = min(spec.from_idx_1, spec.from_idx_2)
    from_idx2 = max(spec.from_idx_1, spec.from_idx_2)

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
        perform_moves_after_splitmerge!(world, moves, (abstract_from_1, abstract_from_2), abstract_changed, index_changed, origin_changed)
    end
end

function perform_moves_after_splitmerge!(world, moves, abstract_from_objects, abstract_changed, index_changed, origin_changed)
    for (_from, _to) in moves
        @assert any(obj in _from.origin for obj in abstract_from_objects) "A move within $spec was specified to move an object $obj which does not have an object being split or merged in its origin!"
        _to = get_or_generate_with_abstract_origin!(world, _to)
        @assert _from.origin != _to.origin "A move within $spec attempted to perform a move which does not change an object's origin."

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
