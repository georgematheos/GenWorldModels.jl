function move_all_between!(world, type; min, inc, max=Inf)
    (new_id_table, changed_ids) = move_all_between(world.id_table, Symbol(type); min=min, inc=inc, max=max)
    world.id_table = new_id_table
    return changed_ids
end

function perform_oupm_update_and_enqueue_downstream!(world, spec)
    changed_ids = perform_oupm_update!(world, spec)
    # TODO: enqueue downstream
end

function perform_oupm_update!(world, spec::BirthMove)
    return move_all_between!(spec.type; min=spec.idx, inc=1)
end
function perform_oupm_update!(world, spec::DeathMove)
    return move_all_between!(spec.type; min=spec.idx, inc=1)
end
function perform_oupm_update!(world, spec::SplitMove)
    from_idx = spec.from_idx
    to_idx1 = min(spec.to_idx1, spec.to_idx2)
    to_idx2 = max(spec.to_idx1, spec.to_idx2)

    changed_ids1, changed_ids2 = nothing, nothing
    if from_idx < to_idx1
        changed_ids1 = move_all_between!(spec.type; min=from_idx, max=to_idx1-1, inc=-1)
        changed_ids2 = move_all_between!(spec.type; min=to_idx2, inc=1)
    elseif from_idx < to_idx2
        changed_ids1 = move_all_between!(spec.type; min=to_idx1, max=from_idx-1, inc=1)
        chagned_ids2 = move_all_between!(spec.type; min=to_idx2, inc=1)
    else
        changed_ids1 = move_all_between!(spec.type; min=to_idx1, max=to_idx2-1, inc=1)
        changed_ids2 = move_all_between!(spec.type; min=to_idx2, max=from_idx-1, inc=2)
    end

    return Iterators.flatten((changed_ids1, changed_ids2))
end
function perform_oupm_update!(world, spec::MergeMove)

end
function perform_oupm_update!(world, spec::MoveMove)

end