# NOTE: The algorithm for updating the world is described and explained in pseudocode
# and in english in `update_algorithm.md`.

####################
# UpdateWorldState #
####################
mutable struct UpdateWorldState <: WorldState
    constraints::ChoiceMap # constraints for updates
    weight::Float64 # total weight of all updates
    discard::DynamicChoiceMap # discard for all world calls
    # the following fields are used for the update algorithm as described in `update_algorithm.md`
    update_queue::PriorityQueue{Call, Int} # queued updates
    fringe_bottom::Int # INVARIANT: every call with topological index < fringe_bottom has been updated if it needs to be
    fringe_top::Int # INVARIANT: no call with topological index > fringe_top has been updated
    visited::Set{Call} # all vertices which we have visited
    calls_whose_dependencies_have_diffs::Set{Call} # any call in this set has had a parent updated without a NoChange diff
    diffs::Dict{Call, Diff} # the diff returned by the update to each call
    call_stack::Stack{Call} # stack of calls we are currently running updates for
    updated_sort_indices::PriorityQueue{Int} # indices the current update cycle will be relaxed into
    call_update_order::Queue{Call} # queue of the new topological order for the current update cycle
    world_update_complete::Bool # once we have finished updating all the calls and move to update the kernel, this goes from false to true
end
function UpdateWorldState(constraints::ChoiceMap)
    UpdateWorldState(
        constraints,
        0., # weight
        choicemap(), # discard
        PriorityQueue{Call, Int}(), # update_queue
        0, 0, # fringe_bottom, fringe_top
        Set{Call}(), Set{Call}(), Dict{Call, Diff}(), # visited, calls_whose_dependencies_have_diffs, diffs
        Stack{Call}(), # call_stack
        PriorityQueue{Int, Int}(), # updated_sort_indices
        Queue{Call}(), # call_update_order
        false # world is up to date
    )
end

##########################
# Diffs for world update #
##########################

"""
    ToBeUpdatedDiff

This diff indicates that a call for the world _may_ have a diff,
since it is yet to be updated.  An update _must_ be called on this call
(via calling `lookup_or_generate!(world, call)`) before the value of this call
after the update can be known.
"""
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

    already_updated = world.state.world_update_complete || (world.call_sort[call] < world.state.fringe_bottom || call in world.state.visited)
    if already_updated # diff is a nochange since nothing was in world.state.diffs
        return NoChange()
    end
    # if we don't yet have a diff, this is a non-updated index, so we don't know
    # how it will change; we will have to update it now.
    return ToBeUpdatedDiff()
end

#################################
# Count and dependency tracking #
#################################

"""
    remove_lookup!(world::World, call::Call, called_from...)

Decrease the count for the number of lookups for `call` in the world.
If another call `called_from` is provided, also decrease
the count for the number of times `call` is looked up by `call_from`.
If decreasing the count brings it to zero, remove the call from the world.

Returns the "discard" due to removals of calls from the world
caused by removing this lookup.  If no call is removed, this is an `EmptyChoiceMap`,
otherwise it is the choicemap for the removed trace.
"""
function remove_lookup!(world::World, call::Call, called_from...)
    new_num_lookups_for_call = note_lookup_removed!(world, call, called_from...)
    if new_num_lookups_for_call == 0
        return remove_call!(world, call)
    else
        return EmptyChoiceMap()
    end
end

"""
    remove_call!(world, call)

Remove the call from the world (ie. remove its subtrace).
Update the world score accordingly, and subtract the removed subtrace's
score from the world.state weight, and add the choices to the state's discard.
Returns the choicemap of the removed trace.
"""
function remove_call!(world, call)
    tr = world.subtraces[call]
    world.subtraces = dissoc(world.subtraces, call)
    score = get_score(tr)
    world.total_score -= score
    world.state.weight -= score
    choices = get_choices(tr)
    set_submap!(world.state.discard, addr(call) => key(call), choices)
    return choices
end

##########################
# World update algorithm #
##########################

"""
    run_world_update!(world)

Entry-point/top level for the world update algorithm.
"""
function run_world_update!(world)
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

"""
    enquque_all_constrained_calls!(world)

Add all the calls for which constraints are provided in `world.state.constraints`
to the `world.state.update_queue`.
"""
function enquque_all_constrained_calls!(world)
    for (mgf_addr, mgf_submap) in get_submaps_shallow(world.state.constraints)
        for (lookup_key, _) in get_submaps_shallow(mgf_submap)
            call = Call(mgf_addr, lookup_key)
            # if there is no value for this call, we'll have to generate it later,
            # so no need to add it to the update queue; generate will get called by the algorithm
            # when it is needed
            if has_value_for_call(world, call)
                enqueue!(world.state.update_queue, call, world.call_sort[call])
            end
        end
    end
end

"""
    update_or_generate!(world, call)

"Visit" the given `call` in the update algorithm by running any necessary
`update` or `generate` call for it.
This function also performs the needed bookkeeping for the update algorithm
to update the dependency structure.
"""
function update_or_generate!(world, call)
    if call in world.state.call_stack
        error("This update will induce a cycle in the topological ordering of the calls in the world!  Call $call will need to have been generated in order to generate itself.")
    end
    
    call_topological_position = world.call_sort[call]
    constraints = get_submap(world.state.constraints, addr(call) => key(call))
    there_are_constraints_for_call = !isempty(constraints) 
    dependency_has_argdiffs = call in world.state.calls_whose_dependencies_have_diffs

    enqueue!(world.state.updated_sort_indices, call_topological_position, call_topological_position)
    world.state.fringe_top = max(world.state.fringe_top, call_topological_position)

    if !has_value_for_call(world, call)
        push!(world.state.call_stack, call)
        run_gen_generate!(world, call, constraints)
        pop!(world.state.call_stack)
        retdiff = UnknownChange()
    elseif there_are_constraints_for_call || dependency_has_argdiffs
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
        enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, call)
    end
    return retdiff
end

"""
    run_gen_update!(world, call, constraints)

Run `Gen.update` for the given call, and update the world
and worldstate accordingly.
"""
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

"""
    run_gen_generate!(world, call, constraints)

Run `Gen.generate` for the given call, and update the world
and worldstate accordingly.
"""
function run_gen_generate!(world, call, constraints)
    weight = generate_value!(world, call, constraints)
    world.state.weight += weight
end

"""
    enqueue_downstream_calls_before_fringe_top!(world, call)

Add every call `v` such that `v` looks up the value to `call`
and such that `v` has topological position < `world.state.fringe_top`
to the `world.state.update_queue`.
"""
function enqueue_downstream_calls_before_fringe_top!(world, call)
    for c in get_all_calls_which_look_up(world, call)
        if world.call_sort[c] < world.state.fringe_top
            if !(c in keys(world.state.update_queue))
                enqueue!(world.state.update_queue, c, world.call_sort[call])
            end
        end
    end
end

"""
    enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, call)

Add every call `v` such that `v` looks up the value to `call`
to the `world.state.update_queue`.
Note in the world state that each of these calls has a dependency
which has an argdiff that is not nochange.
"""
function enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, call)
    for c in get_all_calls_which_look_up(world, call)
        push!(world.state.calls_whose_dependencies_have_diffs, c)
        if !(c in keys(world.state.update_queue))
            enqueue!(world.state.update_queue, c, world.call_sort[call])
        end
    end
end

"""
    reorder_update_cycle!(world)

Change the topological indices of the calls which have been visited
which currently have topological position between `fringe_bottom`
and `fringe_top`.
Do this by filling reassigning the indices in `world.state.updated_sort_indices`
to the calls in `world.state.call_update_order` so that the order
given by `world.state.call_update_order` is preserved in the new index assignment.
"""
function reorder_update_cycle!(world)
    while !isempty(world.state.call_update_order)
        call = dequeue!(world.state.call_update_order)
        index = dequeue!(world.state.updated_sort_indices)
        world.call_sort = change_index_to(world.call_sort, call, index)
    end
    @assert isempty(world.state.updated_sort_indices)
end

##################
# Count updating #
##################

function update_lookup_counts_according_to_world_discard!(world, discard)
    # Note: these calls to `collect` may be important for correctness,
    # because as we remove calls during this function, we call `set_submap!`
    # on the world discard, and so if we have not run `collect`, the lazy iterators
    # returned by get_values_shallow may change!
    # the code is structured so we want this loop to only be over calls in the discard
    # at the time we call this function; if we discard other traces, we will have a seperate
    # call to `update_lookup_counts_according_to_discard!` for this, so if the iterator here
    # also has this call added to it, we will double-count
    for (mgf_addr, mgf_submap) in collect(get_submaps_shallow(discard))
        for (key, submap) in collect(get_submaps_shallow(mgf_submap))
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
            call = Call(mgf_addr, key)
            removed_trace_choicemap = remove_lookup!(world, call, call_discard_comes_from...)
            update_lookup_counts_according_to_discard!(world, removed_trace_choicemap, call)
        else
            update_lookup_counts_according_to_discard!(world, submap, call_discard_comes_from...)
        end
    end
end

"""
    check_no_constrained_calls_deleted(world::World)

Ensure that every call which an update constraint was provided for is still instantiated
in the world after the update.  (Throw an error if this is not the case.)
"""
function check_no_constrained_calls_deleted(world::World)
    for (mgf_addr, mgf_submap) in get_submaps_shallow(world.state.constraints)
        for (key, _) in get_submaps_shallow(mgf_submap)
            call = Call(mgf_addr, key)
            if !has_value_for_call(world, call)
                error("Constraint was provided for $(mgf_addr => key) but this call is not in the world at end of update!")
            end
        end
    end
end

#######################################################
# Interface to `using_world` and `lookup_or_generate` #
#######################################################

"""
    begin_update!(world::World, constraints::ChoiceMap)

Initiate an update for the `UsingWorld` generative function by updating
the calls in the world as specified by `constraints`.
This should be called _before_ updating the `UsingWorld` kernel trace.
"""
function begin_update!(world::World, constraints::ChoiceMap)
    @assert (world.state isa NoChangeWorldState) "cannot initiate an update from a $(typeof(world.state))"
    world.state = UpdateWorldState(constraints)

    # update the world as needed to make it consistent
    # with the `constraints` for the calls we have already generated
    # this will reorder modify topological sort of calls in the world if needed
    run_world_update!(world)

    # note that we are done updating the world, and are
    # now in the process of updating the kernel
    world.state.world_update_complete = true

    if isempty(world.state.visited)
        # if we didn't visit any calls, we didn't even try to update anything!
        return NoChange()
    else
        return WorldUpdateDiff(world)
    end
end

"""
    end_update!(world::World, kernel_discard::ChoiceMap, check_constrained_calls_not_deleted)

End the update process for the world by updating the lookup counts in the world
to remain consistent with the updated world and updated kernel trace.

`check_constrained_calls_not_deleted` is a boolean denoting whether to run a check to confirm
that every call in the world which a constraint was provided for
is still instantiated in the world after this update.
"""
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

"""
    lookup_or_generate_during_update!(world, call)

This should be called when the world is in an update state, and either:
- There is a call to `Gen.generate` for `call`
- There is a call to `Gen.update` for what used to be a different key,
but is being updated so it is now a lookup for `call`
"""
function lookup_or_generate_during_world_update!(world, call)
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

"""
    lookup_or_generate_during_kernel_update!(world, call)

This should be called if, while updating the `UsingWorld` kernel,
there is either a call to `generate` on `lookup_or_generate` for `call`,
or there is an `update` for a `lookup_or_generate` subtrace which changes
the key value so that the new call is `call`.
"""
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
    return get_value_for_call(world, call)
end

function lookup_or_generate_during_update!(world, call)
    if world.state.world_update_complete
        lookup_or_generate_during_kernel_update!(world, call)
    else
        lookup_or_generate_during_world_update!(world, call) 
    end
end