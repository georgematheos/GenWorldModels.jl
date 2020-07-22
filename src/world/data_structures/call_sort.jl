"""
    CallSort

A data structure to store a topological sort for all the calls in the world.
If the algorithms work properly, at the end of every update, the call sort
and the lookup counts should be consistent in that the call sort should
have a valid topological ordering for the dependencies between the calls tracked
in lookup counts.
"""
struct CallSort
    call_to_idx::PersistentHashMap
    max_index::Int
end
CallSort() = CallSort(PersistentHashMap(), 0)
function add_vector_of_new_calls(srt::CallSort, idx_to_call::Vector{Call})
    map = srt.call_to_idx
    for (idx, call) in enumerate(idx_to_call)
        map = assoc(map, call, idx)
    end
    maxidx = max(length(idx_to_call), srt.max_index)
    CallSort(map, maxidx)
end
Base.getindex(srt::CallSort, call::Call) = srt.call_to_idx[call]

function change_index_to(srt::CallSort, call::Call, idx::Int)
    new_map = assoc(srt.call_to_idx, call, idx)
    new_max_idx = max(srt.max_index, idx)
    CallSort(new_map, new_max_idx)
end
add_call_to_end(srt::CallSort, call::Call) = change_index_to(srt, call, srt.max_index + 1)
function remove_call(srt::CallSort, call::Call)
    # note that for performance, we do not go through the calls with index higher than `call`
    # and decrement these indices to make sure the sort's max index stays as small
    # as possible.  Since these are stored as Int64s, we have ~ 2^63
    # updates before we overflow, which practically speaking should be fine
    # for almost any usecase.
    CallSort(
        dissoc(srt.call_to_idx, call),
        srt.max_index
    )
end

# world args and looking up object indices are always before all MGF calls in the sort
# and cannot have their indices changed
const _StaticPositionCall = Union{Call{_world_args_addr}, Call{_get_index_addr}, Call{<:OUPMType{Int}}}
Base.getindex(::CallSort, ::_StaticPositionCall) = -1
change_index_to(srt::CallSort, ::_StaticPositionCall, ::Int) = srt
remove_call(srt::CallSort, call::_StaticPositionCall) = srt