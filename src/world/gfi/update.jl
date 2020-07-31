include("../oupm_updates.jl")

# NOTE: The algorithm for updating the world is described and explained in pseudocode
# and in english in `update_algorithm.md`.

struct WorldUpdateDiff <: Gen.Diff end

####################
# UpdateWorldState #
####################
mutable struct UpdateWorldState <: WorldState
    spec::Gen.UpdateSpec
    externally_constrained_addrs::Selection
    weight::Float64 # total weight of all updates
    discard::DynamicChoiceMap # discard for all world calls
    original_id_table::IDTable # id table from before this update started
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
    # a collection of all the calls which, after being updated, have score=-Inf (ie. probability is 0, so the trace is not in the support)
    # these calls must all be deleted at the end of the update for this update to be valid
    calls_updated_to_unsupported_trace::Set{Call}
    # the choicemap each call which was updated/generated had before the update/generate (if generated, is EmptyChoiceMap)
    original_choicemaps::Dict{Call, ChoiceMap}
    world_update_complete::Bool # once we have finished updating all the calls and move to update the kernel, this goes from false to true
end
function UpdateWorldState(world)
    UpdateWorldState(
        EmptyAddressTree(), # spec
        EmptySelection(), # externally_constrained_addrs
        0., # weight
        choicemap(), # discard
        world.id_table,
        PriorityQueue{Call, Int}(), # update_queue
        0, 0, # fringe_bottom, fringe_top
        Set{Call}(), Set{Call}(), Dict{Call, Diff}(), # visited, calls_whose_dependencies_have_diffs, diffs
        Stack{Call}(), # call_stack
        PriorityQueue{Int, Int}(), # updated_sort_indices
        Queue{Call}(), # call_update_order
        Set{Call}(), # calls_which_must_be_deleted
        Dict{Call, ChoiceMap}(), # original_choicemaps
        false # world is up to date
    )
end

generate_fn(::UpdateWorldState) = Gen.generate

#################################
# Count and dependency tracking #
#################################

"""
    remove_lookup!(world::World, call::Call, called_from...)

Decrease the count for the number of lookups for `call` in the world.
If another call `called_from` is provided, also decrease
the count for the number of times `call` is looked up by `call_from`.
If decreasing the count brings it to zero,
and this is a MGF call (as opposed to a world arg), remove the call from the world.

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
    tr = get_trace(world, call)
    world.traces = dissoc(world.traces, call)
    score = get_score(tr)

    remove_call_from_sort!(world, call)

    if score != -Inf
        world.total_score -= score

        ext_const_addrs = get_subtree(world.state.externally_constrained_addrs, addr(call) => key(call))
        if ext_const_addrs === AllSelection()
            world.state.weight -= score
        else
            world.state.weight -= project(tr, ext_const_addrs)
        end
    else
        # if we are removing a trace with score -Inf, note that we are deleting it
        # no need to update worldscore or weight, since we would have done this in run_gen_update
        # (which should be the only place where 0 probability traces can arise, if UsingWorld is used correctly)
        delete!(world.state.calls_updated_to_unsupported_trace, call)
    end

    if call in keys(world.state.original_choicemaps)
        # if we have updated or generated this call, we have added the original choicemap to `original_choicemaps`
        # so we can discard it
        og_choicemap = world.state.original_choicemaps[call]
    else
        # if there is no map in `original_choicemaps`, it means we haven't changed the trace and are just removing it,
        # so the choicemap is simply that for the currently stored trace
        og_choicemap = get_choices(tr)
    end
    set_submap!(world.state.discard, addr(call) => key(call), og_choicemap)

    # return the choices currently in the deleted trace so we can update counts based on their removal
    return get_choices(tr)
end

remove_call!(_, ::_NonMGFCall) = EmptyAddressTree()


##########################
# World update algorithm #
##########################

"""
    update_world_args_and_enqueue_downstream!(world, new_world_args::NamedTuple, world_argdiffs::NamedTuple)

Updates the world args to the given `new_world_args` if any of the `world_argdiffs` is not
`NoChange()`.  Enqueues all calls which look up the changed args to be updated.
"""
function update_world_args_and_enqueue_downstream!(world, new_world_args, world_argdiffs)
    diffed_addrs = findall(diff -> diff !== NoChange(), world_argdiffs)

    is_diff = false
    for (arg_address, diff) in pairs(world_argdiffs)
        if diff !== NoChange()
            is_diff = true
            call = Call(_world_args_addr, arg_address)
            world.state.diffs[call] = diff
            enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, call)
        end
    end
    if is_diff
        world.world_args = new_world_args
    end
end


"""
    perform_oupm_moves_and_enqueue_downstream!(world, oupm_updates)

Performs all the oupm updates in `oupm_updates` in order, and enqueues
any calls which look at the indices for changed identifiers.
"""
function perform_oupm_moves_and_enqueue_downstream!(world, oupm_updates)
    for oupm_update in oupm_updates
        perform_oupm_move_and_enqueue_downstream!(world, oupm_update)
    end
end

"""
    run_subtrace_updates!(world)

Entry-point/top level for the world update algorithm.
"""
function run_subtrace_updates!(world)
    enquque_all_specd_calls!(world)
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
    enquque_all_specd_calls!(world)

Add all the calls for which an update spec is provided in `world.state.spec`
to the `world.state.update_queue`.
"""
function enquque_all_specd_calls!(world)
    for (mgf_addr, mgf_subspec) in get_subtrees_shallow(world.state.spec)
        if mgf_subspec isa AllSelection
            error("Update specs selecting all calls for a memoized generative function are not currently supported -- this is an implementation TODO!")
        end
        for (lookup_key, _) in get_subtrees_shallow(mgf_subspec)
            call = Call(mgf_addr, lookup_key)
            # if there is no value for this call, we'll have to generate it later,
            # so no need to add it to the update queue; generate will get called by the algorithm
            # when it is needed
            if has_val(world, call) && !haskey(world.state.update_queue, call)
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
    spec = get_subtree(world.state.spec, addr(call) => key(call))
    there_is_spec_for_call = !isempty(spec)
    dependency_has_argdiffs = call in world.state.calls_whose_dependencies_have_diffs

    enqueue!(world.state.updated_sort_indices, call_topological_position, call_topological_position)
    world.state.fringe_top = max(world.state.fringe_top, call_topological_position)

    if !has_val(world, call)
        push!(world.state.call_stack, call)
        run_gen_generate!(world, call, spec)
        pop!(world.state.call_stack)
        retdiff = UnknownChange()
    elseif there_is_spec_for_call || dependency_has_argdiffs
        push!(world.state.call_stack, call)
        ext_const_addrs = get_subtree(world.state.externally_constrained_addrs, addr(call) => key(call))
        retdiff = run_gen_update!(world, call, spec, ext_const_addrs)
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
    run_gen_update!(world, call, spec, externally_constrained_addrs)

Run `Gen.update` for the given call, and update the world
and worldstate accordingly.
"""
function run_gen_update!(world, call, spec, ext_const_addrs)
    old_tr = get_trace(world, call)
    new_tr, weight, retdiff, discard = Gen.update(
        old_tr,
        (world, key(call)),
        (WorldUpdateDiff(), NoChange()),
        spec,
        ext_const_addrs
    )

    if get_score(new_tr) == -Inf
        # if this trace has score -Inf, either the whole world has score -Inf,
        # or it just means that this call has been removed and we're going
        # to delete it at the end of the update. note that we must delete this so we can
        # throw an error if it is not deleted
        push!(world.state.calls_updated_to_unsupported_trace, call)
        
        # we need to subtract off the score for removing this trace here
        # since we won't have access to the old trace later
        score = get_score(old_tr)
        world.state.weight -= score
        world.total_score -= score
    else
        world.state.weight += weight
        world.total_score += get_score(new_tr) - get_score(old_tr)
    end

    world.traces = assoc(world.traces, call, new_tr)
    world.state.original_choicemaps[call] = get_choices(old_tr)
    set_submap!(world.state.discard, addr(call) => key(call), discard)

    if retdiff != NoChange()
        world.state.diffs[call] = retdiff
    end

    return retdiff
end

"""
    run_gen_generate!(world, call, constraints)

Run `Gen.generate` for the given call, and update the world
and worldstate accordingly.
"""
function run_gen_generate!(world, call, constraints)
    weight = generate_value!(world, call, constraints)
    world.state.weight += weight

    # since we're generating this call, there were no choices for it before this generate
    world.state.original_choicemaps[call] = EmptyChoiceMap()
end

function enqueue_call_for_update!(world::World, call::Call)
    if !(call in keys(world.state.update_queue))
        enqueue!(world.state.update_queue, call, world.call_sort[call])
    end
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
            enqueue_call_for_update!(world, c)
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
        enqueue_call_for_update!(world, c)
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

Ensure that every call which an update constraint (ie. an update spec which is not a `Selection`)
was provided for is still instantiated in the world after the update.
(Throw an error if this is not the case.)
"""
function check_no_constrained_calls_deleted(world::World)
    for (mgf_addr, mgf_subspec) in get_subtrees_shallow(world.state.spec)
        for (key, subspec) in get_subtrees_shallow(mgf_subspec)
            if subspec isa Selection
                continue;
            end
            call = Call(mgf_addr, key)
            if !has_val(world, call)
                error("Constraint was provided for $(mgf_addr => key) but this call is not in the world at end of update!")
            end
        end
    end
end

#######################################################
# Interface to `using_world` and `lookup_or_generate` #
#######################################################

"""
    begin_update!(world::World, oupm_moves::Tuple, new_world_args::NamedTuple, world_argdiffs::NamedTuple)

Initiate an update for the `UsingWorld` generative function.  This will shallow-copy
the world and put the new copy in update mode.  It will update the world args
as specified, and perform the specified open universe moves, then return
the new world object.
This will not yet update any memoized generative function calls.
"""
function begin_update(world::World, oupm_moves::Tuple, new_world_args::NamedTuple, world_argdiffs::NamedTuple)
    @assert (world.state isa NoChangeWorldState) "cannot initiate an update from a $(typeof(world.state))"

    # shallow copy the world and put in update mode
    world = World(world)
    world.state = UpdateWorldState(world)

    # update the world args, and enqueue all the
    # calls which look up these world args to be updated
    update_world_args_and_enqueue_downstream!(world, new_world_args, world_argdiffs)

    # perform the moves specified in `oupm_moves`, and enqueue all calls
    # which look up the indices for these and hence need to be updated
    perform_oupm_moves_and_enqueue_downstream!(world, oupm_moves)

    return world
end

"""
    update_mgf_calls!(world::World, spec, ext_const_addrs)

Update all the calls in the `world` according to the given update spec,
with weight determined by the given `ext_const_addrs`.
"""
function update_mgf_calls!(world::World, spec::Gen.UpdateSpec, ext_const_addrs::Gen.Selection)
    world.state.spec = spec
    world.state.externally_constrained_addrs = ext_const_addrs

    # update the world as needed to make it consistent
    # with the `specs` for the calls we have already generated
    # this will reorder modify topological sort of calls in the world if needed
    run_subtrace_updates!(world)

    # note that we are done updating the world, and are
    # now in the process of updating the kernel
    world.state.world_update_complete = true
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

    if !isempty(world.state.calls_updated_to_unsupported_trace)
        invalid_calls = collect(world.state.calls_updated_to_unsupported_trace)
        error("""This update appears to have caused the world to have score -Inf!
        In particular, the update caused the scores for the traces for calls $invalid_calls
        to have score -Inf, and these calls were not deleted from the world.
        """)
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

function lookup_or_generate_during_world_update!(world, call, reason_for_call)
    no_val = !has_val(world, call)
    if no_val && is_mgf_call(call)
        # add this call to the sort, at the end for now
        world.call_sort = add_call_to_end(world.call_sort, call)
    end

    # if we haven't yet handled the update for this call, we have to do that before proceeding
    if no_val || (world.call_sort[call] >= world.state.fringe_bottom && !(call in world.state.visited))
        update_or_generate!(world, call)
    end

    # if we `generate`d, or did a key change update, this is a new lookup
    # (if on the other hand this update occurred due to a `ToBeUpdatedDiff`, it is not a new lookup)
    if reason_for_call == :generate || reason_for_call == :key_change
        note_new_lookup!(world, call, world.state.call_stack)    
    end

    return get_val(world, call)
end

"""
    lookup_or_generate_during_kernel_update!(world, call)

This should be called if, while updating the `UsingWorld` kernel,
there is either a call to `generate` on `lookup_or_generate` for `call`,
or there is an `update` for a `lookup_or_generate` subtrace which changes
the key value so that the new call is `call`.
"""
function lookup_or_generate_during_kernel_update!(world, call)
    if !has_val(world, call)
        spec = get_subtree(world.state.spec, addr(call) => key(call))
        push!(world.state.call_stack, call)
        weight = generate_value!(world, call, spec)
        pop!(world.state.call_stack)
        world.call_sort = add_call_to_end(world.call_sort, call)
        world.state.weight += weight
    end
    note_new_lookup!(world, call, world.state.call_stack)
    return get_val(world, call)
end

"""
    lookup_or_generate_during_update!(world, call)

This should be called when the world is in an update state, and one of the following occurs:
- There is a call to `Gen.generate` for `call`; in this case we should have `reason_for_call = :generate`
- There is a call to `Gen.update` for what used to be a different key,
but is being updated so it is now a lookup for `call`.  In this case we should have
`reason_for_call = :key_change`
- There is a call to `Gen.update` for a lookup for a key which is diffed with a `ToBeUpdatedDiff`
in the world.  In this case we should have `reason_for_call = :to_be_updated`
"""
function lookup_or_generate_during_update!(world, call, reason_for_call)
    if world.state.world_update_complete
        lookup_or_generate_during_kernel_update!(world, call)
    else
        lookup_or_generate_during_world_update!(world, call, reason_for_call) 
    end
end