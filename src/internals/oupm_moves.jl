abstract type OUPMMove end

"""
    Move(from::ConcreteIndexOUPMObject, to::ConcreteIndexOUPMObject)

An object manipulation move from `from` to `to`.
"""
struct Move{T} <: OUPMMove
    from::ConcreteIndexOUPMObject{T}
    to::ConcreteIndexOUPMObject{T}
end

"""
    Create(object::ConcreteIndexOUPMObject)

An object manipulation move to create a new object with the given type, index, and origin.
"""
struct Create{T} <: OUPMMove
    obj::ConcreteIndexOUPMObject{T}
end

"""
    Delete(object::ConcreteIndexOUPMObject)

An object manipulation move to create a new object with the given type, index, and origin.
"""
struct Delete{T} <: OUPMMove
    obj::ConcreteIndexOUPMObject{T}
end

"""
    Split(from::ConcreteIndexOUPMObject, to_idx_1::Integer, to_idx_2::Integer; moves=())

An object manipulation move to split `from` into 2 objects with indices `to_idx_1` and `to_idx_2`.
`moves` is a tuple of pairs `prev => new`, where `prev` is an object which had `from` as an origin,
and `new` is a new object of the same type of `prev` with the new origin and index `prev` should be given.
`new` may also be `nothing` if `prev` should be deleted.
"""
struct Split{T} <: OUPMMove
    from::ConcreteIndexOUPMObject{T}
    to_idx_1::Int
    to_idx_2::Int
    moves::Tuple{Vararg{<:Pair{<:ConcreteIndexOUPMObject, <:Union{Nothing, <:ConcreteIndexOUPMObject}}}}
end
Split(from, to_idx_1::Integer, to2::ConcreteIndexOUPMObject, moves) = Split(from, to_idx_1, _get_idx(to2, from), moves)
Split(from, to1::ConcreteIndexOUPMObject, to2, moves) = Split(from, _get_idx(to1, from), to2, moves)
Split(from, to1, to2; moves=()) = Split(from, to1, to2, moves)

"""
    Merge(to::ConcreteIndexOUPMObject, from_idx_1, from_idx_2; moves=())

An object manipulation move to merge the objects with the type and origin of `to` at indices
`from_idx_1` and `from_idx_2` to the index of `to`.

`moves` is a tuple of pairs `prev => new`, where `prev` is an object which had one of the objects being merged
 as an origin, and `new` is a new object of the same type of `prev` with the new origin and index `prev`
 should be given.
`new` may also be `nothing` if `prev` should be deleted.
"""
struct Merge{T} <: OUPMMove
    to::ConcreteIndexOUPMObject{T}
    from_idx_1::Int
    from_idx_2::Int
    moves::Tuple{Vararg{<:Pair{<:ConcreteIndexOUPMObject, <:Union{Nothing, <:ConcreteIndexOUPMObject}}}}
end
function _get_idx(obj::ConcreteIndexOUPMObject, check_origin_type_matches)
    @assert obj.origin == check_origin_type_matches.origin
    @assert oupm_type_name(obj) == oupm_type_name(check_origin_type_matches)
    return obj.idx
end
Merge(to, from1::Integer, from2::ConcreteIndexOUPMObject, moves) = Merge(to, from1, _get_idx(from2, to), moves)
Merge(to, from1::ConcreteIndexOUPMObject, from2, moves) = Merge(to, _get_idx(from1, to), from2, moves)
Merge(to, from1, from2; moves=()) = Merge(to, from1, from2, moves)

"""
    WorldUpdate(object_moves::Tuple{Vararg{<:OUPMMove}}, subspec)
    WorldUpdate(object_moves..., subspec)
    WorldUpdate(object_moves...)

An update specification for a `UsingWorld` trace which applies the given
`object_moves` in order, then updates the trace via the given `subspec`.
If no `subspec` is provided, the update only performs the given object manipulation moves.
"""
struct WorldUpdate <: Gen.CustomUpdateSpec
    moves::Tuple{Vararg{<:OUPMMove}}
    subspec::Gen.UpdateSpec
    WorldUpdate(moves::Tuple{Vararg{<:OUPMMove}}, subspec) = new(moves, subspec)
end
function WorldUpdate(args...)
    if args[end] isa Gen.AddressTree
        WorldUpdate(args[1:end-1], args[end])
    else
        WorldUpdate(args, EmptyAddressTree())
    end
end

function GenTraceKernelDSL.undualize(spec::WorldUpdate)
    WorldUpdate(spec.moves, GenTraceKernelDSL.undualize(spec.subspec))
end
function GenTraceKernelDSL.incorporate_regenerate_constraints!(
    spec::WorldUpdate, trace_choices::ChoiceMap, reverse_regenerated::Selection
)
    WorldUpdate(
        spec.moves,
        GenTraceKernelDSL.incorporate_regenerate_constraints!(spec.subspec, trace_choices, reverse_regenerated)
    )
end
function GenTraceKernelDSL.dualized_values(spec::WorldUpdate)
    GenTraceKernelDSL.dualized_values(spec.subspec)
end

function reverse_moves(moves::Tuple{Vararg{<:OUPMMove}})
    Tuple(reverse_move(moves[j]) for j=length(moves):-1:1)
end

reverse_move(m::Move) = Move(m.to, m.from)
reverse_move(m::Create) = Delete(m.obj)
reverse_move(m::Delete) = Create(m.obj)
function reverse_move(m::Split)
    revmoves = map(swappair, filter(x -> x[2] !== nothing, m.moves))
    Merge(m.from, m.to_idx_1, m.to_idx_2, revmoves)
end
function reverse_move(m::Merge)
    revmoves = map(swappair, filter(x -> x[2] !== nothing, m.moves))
    Split(m.to, m.from_idx_1, m.from_idx_2, revmoves)
end

swappair((x, y)) = y => x