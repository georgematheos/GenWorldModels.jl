abstract type OUPMMove end

struct MoveMove{T} <: OUPMMove
    from::ConcreteIndexOUPMObject{T}
    to::ConcreteIndexOUPMObject{T}
end

struct UpdateWithOUPMMovesSpec <: Gen.CustomUpdateSpec
    moves::Tuple{Vararg{<:OUPMMove}}
    subspec::Gen.UpdateSpec
end

export MoveMove #, BirthMove, DeathMove, SplitMove, MergeMove
export UpdateWithOUPMMovesSpec

function reverse_moves(moves::Tuple)
    Tuple(reverse_move(moves[j]) for j=length(moves):-1:1)
end

reverse_move(m::MoveMove) = MoveMove(m.to, m.from)

# reverse_move(m::BirthMove) = DeathMove(m.type, m.idx)
# reverse_move(m::DeathMove) = BirthMove(m.type, m.idx)
# reverse_move(m::SplitMove) = MergeMove(m.type, m.from_idx, m.to_idx1, m.to_idx2)
# reverse_move(m::MergeMove) = SplitMove(m.type, m.to_idx, m.from_idx1, m.from_idx2)


# struct BirthMove <: OUPMMove
#     type::Type{<:OUPMType}
#     idx::Int
# end
# struct DeathMove <: OUPMMove
#     type::Type{<:OUPMType}
#     idx::Int
# end
# struct MergeMove <: OUPMMove
#     type::Type{<:OUPMType}
#     to_idx::Int
#     from_idx1::Int
#     from_idx2::Int
# end
# struct SplitMove <: OUPMMove
#     type::Type{<:OUPMType}
#     from_idx::Int
#     to_idx1::Int
#     to_idx2::Int
# end