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
function get_diff_for_index(diff::WorldUpdateDiff, address::CallAddr)
    WorldUpdateAddrDiff(diff.world, address)
end

"""
    WorldUpdateAddrDiff

An indexdiff containing information about what values for a certain memoized generative
function address have changed in the world during an update.
"""
struct WorldUpdateAddrDiff <: IndexDiff
    world::World
    address::CallAddr
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
