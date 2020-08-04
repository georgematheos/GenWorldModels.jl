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

        concrete_obj = world.id_table[abstract]
        if !haskey(original_id_table, concrete_obj)
            note_new_call!(world, Call(_get_abstract_addr, concrete_obj))
        end
    end
    for abstract in origin_changed
        c1 = Call(_get_origin_addr, abstract)
        c2 = Call(_get_concrete_addr, abstract)
        world.state.diffs[c1] = UnknownChange()
        world.state.diffs[c2] = UnknownChange()
        enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, c1)
        enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, c2)

        concrete_obj = world.id_table[abstract]
        if !haskey(original_id_table, concrete_obj)
            note_new_call!(world, Call(_get_abstract_addr, concrete_obj))
        end
    end
    for concrete in abstract_changed
        c = Call(_get_abstract_addr, concrete)
        world.state.diffs[c] = UnknownChange()
        enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, c)
    end
end

function move_all_between!(world::World, typename::Symbol, origin::Tuple{Vararg{<:AbstractOUPMObject}}, index_changed, abstract_changed; min=1, inc, max=Inc)
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
    @assert world.id_table[from] != world.id_table[to]
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
        move_all_between!(world, typename(from), from.origin, index_changed, abstract_changed; max=max, min=min, inc=inc)
    else
        # move down all above `from`
        move_all_between!(world, typename(from), from.origin, index_changed, abstract_changed; min=from.idx+1, max=Inf, inc=-1)
        # move up all after and including `to`
        move_all_between!(world, typename(to), to.origin, index_changed, abstract_changed; min=to.idx, max=Inf, inc=+1)

        if has_abstract
            push!(origin_changed, abstract)
        end
    end
    if has_abstract
        world.id_table = assoc(world.id_table, abstract, to)
    end
end



# function perform_oupm_move!(world, spec::BirthMove)
#     return move_all_between!(world, spec.type; min=spec.idx, inc=1)
# end
# function perform_oupm_move!(world, spec::DeathMove)
#     return move_all_between!(world, spec.type; min=spec.idx+1, inc=-1)
# end
# function perform_oupm_move!(world, spec::SplitMove)
#     from_idx = spec.from_idx
#     to_idx1 = min(spec.to_idx1, spec.to_idx2)
#     to_idx2 = max(spec.to_idx1, spec.to_idx2)

#     # capture the variables common to all the `move_all_between!` calls we need, to simplify syntax
#     changed_indices = Set{UUID}()
#     mb! = let w=world, c=changed_indices, t=spec.type
#         (min, max, inc) -> union!(c, move_all_between!(w, t; min=min, max=max, inc=inc))
#     end

#     if from_idx < to_idx1
#         mb!(from_idx + 1, to_idx1, -1),
#         mb!(to_idx2, Inf, +1)
#     elseif from_idx < to_idx2
#         mb!(to_idx1, from_idx - 1, +1),
#         mb!(to_idx2, Inf, +1)
#     else
#         mb!(from_idx + 1, Inf, +1),
#         mb!(to_idx2 - 1, from_idx - 1, +2),
#         mb!(to_idx1, to_idx2 - 2, +1)
#     end

#     return changed_indices
# end
# function perform_oupm_move!(world, spec::MergeMove)
#     to_idx = spec.to_idx
#     from_idx1 = min(spec.from_idx1, spec.from_idx2)
#     from_idx2 = max(spec.from_idx1, spec.from_idx2)

#     # capture the variables common to all the `move_all_between!` calls we need, to simplify syntax
#     changed_indices = Set{UUID}()
#     mb! = let w=world, c=changed_indices, t=spec.type
#         (min, max, inc) -> union!(c, move_all_between!(w, t; min=min, max=max, inc=inc))
#     end

#     if to_idx < from_idx1
#         mb!(from_idx2 + 1, Inf, -1),
#         mb!(to_idx, from_idx1 - 1, +1)
#     elseif to_idx < from_idx2
#         mb!(from_idx2 + 1, Inf, -1),
#         mb!(from_idx1 + 1, to_idx, -1)
#     else
#         mb!(from_idx1 + 1, from_idx2 - 1, -1),
#         mb!(from_idx2 + 1, to_idx + 1, -2),
#         mb!(to_idx + 2, Inf, -1)
#     end

#     return changed_indices
# end