function move_all_between!(world, changed_ids, type; min, inc, max=Inf)
    new_id_table = move_all_between(world.id_table, changed_ids, oupm_type_name(type); min=min, inc=inc, max=max)
    world.id_table = new_id_table
end

function perform_oupm_move_and_enqueue_downstream!(world, spec)
    changed_ids = perform_oupm_move!(world, spec)
    for id in changed_ids
        # note there is a diff for looking up the index; enqueue calls that use this idx to be updated
        call = Call(_get_index_addr, spec.type(id))
        world.state.diffs[call] = UnknownChange()
        enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, call)
    end
end

function perform_oupm_move!(world, spec::BirthMove)
    changed_ids = Set{UUID}()
    move_all_between!(world, changed_ids, spec.type; min=spec.idx, inc=1)
    return changed_ids
end
function perform_oupm_move!(world, spec::DeathMove)
    changed_ids = Set{UUID}()
    move_all_between!(world, changed_ids, spec.type; min=spec.idx, inc=-1)
    return changed_ids
end
function perform_oupm_move!(world, spec::SplitMove)
    from_idx = spec.from_idx
    to_idx1 = min(spec.to_idx1, spec.to_idx2)
    to_idx2 = max(spec.to_idx1, spec.to_idx2)

    changed_ids = Set{UUID}()
    # capture the variables common to all the `move_all_between!` calls we need, to simplify syntax
    mb! = let w=world, c=changed_ids, t=spec.type
        (min, max, inc) -> move_all_between!(w, c, t; min=min, max=max, inc=inc)
    end

    if from_idx < to_idx1
        mb!(from_idx + 1, to_idx1, -1)
        mb!(to_idx2, Inf, +1)
    elseif from_idx < to_idx2
        mb!(to_idx1, from_idx - 1, +1)
        mb!(to_idx2, Inf, +1)
    else
        mb!(from_idx + 1, Inf, +1)
        mb!(to_idx2 - 1, from_idx - 1, +2)
        mb!(to_idx1, to_idx2 - 2, +1)
    end

    return changed_ids
end
function perform_oupm_move!(world, spec::MergeMove)
    to_idx = spec.to_idx
    from_idx1 = min(spec.from_idx1, spec.from_idx2)
    from_idx2 = max(spec.from_idx1, spec.from_idx2)

    changed_ids = Set{UUID}()
    # capture the variables common to all the `move_all_between!` calls we need, to simplify syntax
    mb! = let w=world, c=changed_ids, t=spec.type
        (min, max, inc) -> move_all_between!(w, c, t; min=min, max=max, inc=inc)
    end

    if to_idx < from_idx1
        mb!(from_idx2 + 1, Inf, -1)
        mb!(to_idx, from_idx1 - 1, +1)
    elseif to_idx < from_idx2
        mb!(from_idx2 + 1, Inf, -1)
        mb!(from_idx1 + 1, to_idx, -1)
    else
        mb!(from_idx1 + 1, from_idx2 - 1, -1)
        mb!(from_idx2 + 1, to_idx + 1, -2)
        mb!(to_idx + 2, Inf, -1)
    end
    
    return changed_ids
end
function perform_oupm_move!(world, spec::MoveMove)
    typename = oupm_type_name(spec.type)
    
    changed_ids = Set{UUID}()
    _has_id = has_idx(world.id_table, typename, spec.from_idx)

    if _has_id
        id = get_id(world.id_table, spec.type, spec.from_idx)
    end
    if spec.from_idx < spec.to_idx
        move_all_between!(world, changed_ids, spec.type; min=spec.from_idx+1, max=spec.to_idx, inc=-1)
    else
        move_all_between!(world, changed_ids, spec.type; min=spec.to_idx, max=spec.from_idx-1, inc=+1)
    end
    if _has_id
        world.id_table = insert_id(world.id_table, typename, spec.to_idx, id)
        push!(changed_ids, id)
    end

    return changed_ids
end