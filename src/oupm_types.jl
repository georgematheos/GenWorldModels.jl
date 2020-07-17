abstract type OUPMType{T} end
macro type(name::Symbol)
    quote
        struct $(esc(name)){T} <: OUPMType{T}
            idx_or_id::T
            $(esc(name))(x::Int) = new{Int}(x)
            $(esc(name))(x::UUID) = new{UUID}(x)
        end
    end
end

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