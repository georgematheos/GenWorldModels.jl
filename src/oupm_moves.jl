abstract type OUPMMove end

struct MoveMove{T} <: OUPMMove
    from::ConcreteIndexOUPMObject{T}
    to::ConcreteIndexOUPMObject{T}
end

struct BirthMove{T} <: OUPMMove
    obj::ConcreteIndexOUPMObject{T}
end
struct DeathMove{T} <: OUPMMove
    obj::ConcreteIndexOUPMObject{T}
end

struct SplitMove{T} <: OUPMMove
    from::ConcreteIndexOUPMObject{T}
    to_idx_1::Int
    to_idx_2::Int
    moves::Tuple{Vararg{<:Pair{<:ConcreteIndexOUPMObject, <:Union{Nothing, <:ConcreteIndexOUPMObject}}}}
end
SplitMove(from, to_idx_1, to_idx_2; moves=()) = SplitMove(from, to_idx_1, to_idx_2, moves)
struct MergeMove{T} <: OUPMMove
    to::ConcreteIndexOUPMObject{T}
    from_idx_1::Int
    from_idx_2::Int
    moves::Tuple{Vararg{<:Pair{<:ConcreteIndexOUPMObject, <:Union{Nothing, <:ConcreteIndexOUPMObject}}}}
end
MergeMove(to, from_idx_1, from_idx_2; moves=()) = MergeMove(to, from_idx_1, from_idx_2, moves)

struct UpdateWithOUPMMovesSpec <: Gen.CustomUpdateSpec
    moves::Tuple{Vararg{<:OUPMMove}}
    subspec::Gen.UpdateSpec
end

export MoveMove, BirthMove, DeathMove, SplitMove, MergeMove
export UpdateWithOUPMMovesSpec

function reverse_moves(moves::Tuple{Vararg{<:OUPMMove}})
    Tuple(reverse_move(moves[j]) for j=length(moves):-1:1)
end

reverse_move(m::MoveMove) = MoveMove(m.to, m.from)
reverse_move(m::BirthMove) = DeathMove(m.obj)
reverse_move(m::DeathMove) = BirthMove(m.obj)
function reverse_move(m::SplitMove)
    revmoves = map(swappair, filter(x -> x[2] !== nothing, m.moves))
    MergeMove(m.from, m.to_idx_1, m.to_idx_2, revmoves)
end
function reverse_move(m::MergeMove)
    revmoves = map(swappair, filter(x -> x[2] !== nothing, m.moves))
    SplitMove(m.to, m.from_idx_1, m.from_idx_2, revmoves)
end

swappair((x, y)) = y => x