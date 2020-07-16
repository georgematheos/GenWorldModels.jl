abstract type OUPMType end
(T::Type{<:OUPMType})(world, idx::Int) = lookup_or_generate_id_object(world, T, idx)

abstract type OUPMMove end
struct BirthMove <: OUPMMove
    type::Type{<:OUPMType}
    idx::Int
end
struct DeathMove <: OUPMMove
    type::Type{<:OUPMType}
    idx::Int
end
struct MergeMove <: OUPMMove
    type::Type{<:OUPMType}
    to_idx::Int
    from_idx1::Int
    from_idx2::Int
end
struct SplitMove <: OUPMMove
    type::Type{<:OUPMType}
    from_idx::Int
    to_idx1::Int
    to_idx2::Int
end
struct MoveMove <: OUPMMove
    type::Type{<:OUPMType}
    from_idx::Int
    to_idx::Int
end

struct UpdateWithOUPMMovesSpec <: Gen.CustomUpdateSpec
    moves::Tuple{Vararg{<:OUPMMove}}
    subspec::Gen.UpdateSpec
end