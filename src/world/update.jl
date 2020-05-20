### UpdateWorldState ###
mutable struct UpdateWorldState <: WorldState
    constraints::ChoiceMap # constraints for updates
    weight::Float64 # total weight of all updates
    discard::DynamicChoiceMap # discard for all world calls
    # the following fields are used for the update algorithm as described in `update_algorithm.md`
    update_queue::PriorityQueue{Call, Int} # queued updates
    fringe_bottom::Int # INVARIANT: every call with topological index < fringe_bottom has been updated if it needs to be
    fringe_top::Int # INVARIANT: no call with topological index > fringe_top has been updated
    visited::Set{Call} # all vertices which we have visited
    diffs::Dict{Call, Diff} # the diff returned by the update to each call
    call_stack::Stack{Call} # stack of calls we are currently running updates for
    updated_sort_indices::PriorityQueue{Int} # indices the current update cycle will be relaxed into
    call_update_order::Queue{Call} # queue of the new topological order for the current update cycle
    world_call_updates_complete::Bool # once we have finished updating all the calls and move to update the kernel, this goes from false to true
end
function UpdateWorldState(constraints::ChoiceMap)
    UpdateWorldState(
        constraints,
        0., # weight
        choicemap(), # discard
        PriorityQueue{Call, Int}(), # update_queue
        0, 0, # fringe_bottom, fringe_top
        Set{Call}(), Dict{Call, Diff}(), # visited, diffs
        Stack{Call}(), # call_stack
        PriorityQueue{Int, Int}(), # updated_sort_indices
        Queue{Call}(), # call_update_order
        false # world is up to date
    )
end

### Diffs for world update ###

# denotes that this call in a world needs to be updated
struct ToBeUpdatedDiff <: Gen.Diff end
"""
    WorldUpdateDiff <: IndexDiff

An indexdiff containing information about what values in the world have changed during
an update.
"""
struct WorldUpdateDiff <: IndexDiff
    world::World
end
function get_diff_for_index(diff::WorldUpdateDiff, address)
    WorldUpdateAddrDiff(diff.world, address)
end
"""
    WorldUpdateAddrDiff

An indexdiff containing information about what values for a certain memoized generative
function address have changed in the world during an update.
"""
struct WorldUpdateAddrDiff <: IndexDiff
    world::World
    address::Symbol
end
function get_diff_for_index(diff::WorldUpdateAddrDiff, key)
    call = Call(diff.address, key)
    world = diff.world

    if haskey(world.state.diffs, call)
        return world.state.diffs[call]
    end

    already_updated = world.state.world_call_updates_complete || (world.call_sort[call] < world.state.fringe_bottom || call in world.state.visited)
    if already_updated # diff is a nochange since nothing was in world.state.diffs
        return NoChange()
    end
    # if we don't yet have a diff, this is a non-updated index, so we don't know
    # how it will change; we will have to update it now.
    return ToBeUpdatedDiff()
end

### Count and dependency tracking ###

function remove_lookup!(world::World, call::Call, called_from...)
    new_num_lookups_for_call = note_lookup_removed!(world, call, called_from...)
    if new_num_lookups_for_call == 0
        remove_call!(world, call)
    end
end

function remove_call!(world, call)
    tr = world.subtraces[call]
    world.subtraces = dissoc(world.subtraces, call)
    score = get_score(tr)
    world.total_score -= score
    world.state.weight -= score
end

### Update algorithm ###

function run_updates_from_constraints_or_selections!(world)
    enquque_all_constrained_calls!(world)
    while !isempty(world.state.update_queue)
        call = dequeue!(world.state.update_queue)
        if !(call in world.state.visited)
            if world.call_sort[call] > world.state.fringe_top
                reorder_update_cycle!(world)
                fringe_top = world.call_sort[call]
            end
            world.state.fringe_bottom = world.call_sort[call]
            update_or_generate!(world, call)
        end
    end
end

function enquque_all_constrained_calls!(world)
    for (mgf_addr, mgf_submap) in get_submaps_shallow(world.state.constraints)
        for (lookup_key, _) in get_submaps_shallow(mgf_submap)
            call = Call(mgf_addr, lookup_key)
            enqueue!(world.state.update_queue, call, world.call_sort[call])
        end
    end
end

function update_or_generate!(world, call)
    if call in world.state.call_stack
        error("This update will induce a cycle in the topological ordering of the calls in the world!  Call $call will need to have been generated in order to generate itself.")
    end
    call_topological_position = world.call_sort[call]
    constraints = get_submap(world.state.constraints, addr(call) => key(call))
    there_are_constraints_for_call = !isempty(constraints) 
    there_is_argdiff_for_call = haskey(world.state.diffs, call)

    enqueue!(world.state.updated_sort_indices, call_topological_position, call_topological_position)

    if !has_value_for_call(world, call)
        push!(world.state.call_stack, call)
        run_gen_generate!(world, call, constraints)
        pop!(world.state.call_stack)
        retdiff = UnknownChange()
    elseif there_are_constraints_for_call || there_is_argdiff_for_call
        push!(world.state.call_stack, call)
        retdiff = run_gen_update!(world, call, constraints)
        pop!(world.state.call_stack)
    else
        retdiff = NoChange()
    end
    enqueue!(world.state.call_update_order, call)
    push!(world.state.visited, call)
    if retdiff == NoChange()
        enqueue_downstream_calls_before_fringe_top!(world, call)
    else
        enqueue_all_downstream_calls!(world, call)
    end
    return retdiff
end

function run_gen_update!(world, call, constraints)
    old_tr = world.subtraces[call]
    new_tr, weight, retdiff, discard = Gen.update(
        old_tr,
        (world, key(call)),
        (WorldUpdateDiff(world), NoChange()),
        constraints
    )
    world.subtraces = assoc(world.subtraces, call, new_tr)
    world.state.weight += weight
    world.total_score += get_score(new_tr) - get_score(old_tr)
    if retdiff != NoChange()
        world.state.diffs[call] = retdiff
    end
    set_submap!(world.state.discard, addr(call) => key(call), discard)
end

function run_gen_generate!(world, call, constraints)
    weight = generate_value!(world, call, constraints)
    world.state.weight += weight
end

function enqueue_downstream_calls_before_fringe_top!(world, call)
    for c in get_calls_looked_up_in(world, call)
        if world.call_sort[c] < world.state.fringe_top
            enqueue!(world.state.update_queue, c)
        end
    end
end

function enqueue_all_downstream_calls!(world, call)
    for c in get_calls_looked_up_in(world, call)
        enqueue!(world.state.update_queue, c)
    end
end

function reorder_update_cycle!(world)
    while !isempty(world.state.call_update_order)
        call = dequeue!(world.state.call_update_order)
        index = dequeue!(world.state.updated_sort_indices)
        world.call_sort = change_index_to(world.call_sort, call, index)
    end
    @assert isempty(world.state.updated_sort_indices)
end

function update_lookup_counts_according_to_world_discard!(world, discard)
    for (mgf_addr, mgf_submap) in get_values_shallow(discard)
        for (key, submap) in get_values_shallow(mgf_submap)
            call_submap_discard_is_for = Call(mgf_addr, key)
            update_lookup_counts_according_to_discard!(world, submap, call_submap_discard_is_for)
        end
    end
end

function update_lookup_counts_according_to_kernel_discard!(world, kernel_discard)
    update_lookup_counts_according_to_discard!(world, kernel_discard)
end

function update_lookup_counts_according_to_discard!(world, discard, call_discard_comes_from...)
    for (address, submap) in get_submaps_shallow(discard)
        if address == metadata_addr(world)
            (mgf_addr, key) = first(get_values_shallow(submap))
            remove_lookup!(world, Call(mgf_addr, key), call_discard_comes_from...)
        else
            update_lookup_counts_according_to_discard!(world, submap, call_discard_comes_from...)
        end
    end
end

function check_no_constrained_calls_deleted(world::World)
    for (mgf_addr, mgf_submap) in get_submaps_shallow(world.state.constraints)
        for (key, _) in get_submaps_shallow(mgf_submap)
            call = Call(mgf_addr, key)
            if !has_value_for_call(world, call)
                error("Constraint was provided for $(mgf_addr => key) but this call was deleted from the world!")
            end
        end
    end
end

### Interface to `using_world` and `lookup_or_generate` ###

function begin_update!(world::World, constraints::ChoiceMap)
    @assert (world.state isa NoChangeWorldState) "cannot initiate an update from a $(typeof(world.state))"
    world.state = UpdateWorldState(constraints)

    # update the world as needed to make it consistent
    # with the `constraints` for the calls we have already generated
    # this will reorder modify topological sort of calls in the world if needed
    run_updates_from_constraints_or_selections!(world)
    world.state.world_call_updates_complete = true

    return WorldUpdateDiff(world)
end

function end_update!(world::World, kernel_discard::ChoiceMap, check_constrained_calls_not_deleted)
    update_lookup_counts_according_to_world_discard!(world, world.state.discard)
    update_lookup_counts_according_to_kernel_discard!(world, kernel_discard)
    retval = (world.state.weight, world.state.discard)

    if check_constrained_calls_not_deleted
        check_no_constrained_calls_deleted(world)
    end

    # I thought about running a check that there are no cycles in the world after the update,
    # in case a memoized generative function has a faulty update function which
    # doesn't run `update` on all `lookup_or_generate` calls in it,
    # so we don't detect a cycle being created during the update itself,
    # but the check during `generate` should already ensure that all `lookup_or_generate`
    # calls are traced, and I think we should assume all generative functions' `update` methods
    # work properly, so I think we don't need this check here

    world.state = NoChangeWorldState()

    return retval
end

# during an update, this should be called if there was a call to `update(lookup_or_generate_tr, call)`
# where `lookup_or_generate_tr` was previously looking up a different key,
# or where `lookup_or_generate_tr` has not had its key change, but has a key which
# appears topologically AFTER fringe_bottom and so is part of the current "update cycle",
# or if there was a call to `generate(lookup_or_generate_tr, call)`
"""
    lookup_or_generate_during_update!(world, call)

This should be called when the world is in an update state, and either:
- There is a call to `Gen.generate` for `call`
- There is a call to `Gen.update` for what used to be a different key,
but is being updated so it is now a lookup for `call`
"""
function lookup_or_generate_during_world_call_updates!(world, call)
    if !has_value_for_call(world, call)
        # add this call to the sort, at the end for now
        world.call_sort = add_call_to_end(world.call_sort, call)
    end

    # if we haven't yet handled the update for this call, we have to do that before proceeding
    if world.call_sort[call] >= world.state.fringe_bottom && !(call in world.state.visited)
        update_or_generate!(world, call)
    end

    note_new_lookup!(world, call, world.state.call_stack)    

    return get_value_for_call(world, call)
end

# this should only ever be called during a generate call
function lookup_or_generate_during_kernel_update!(world, call)
    if !has_value_for_call(world, call)
        constraints = get_submap(world.state.constraints, addr(call) => key(call))
        push!(world.state.call_stack, call)
        weight = generate_value!(world, call, constraints)
        pop!(world.state.call_stack)
        world.call_sort = add_call_to_end(world.call_sort, call)
        world.state.weight += weight
    end
    note_new_lookup!(world, call, world.state.call_stack)
    return get_retval(world.subtraces[call])
end

function lookup_or_generate_during_update!(world, call)
    if world.state.world_call_updates_complete
        lookup_or_generate_during_kernel_update!(world, call)
    else
        lookup_or_generate_during_world_call_updates!(world, call) 
    end
end