"""
    OUPMType{UUID}

Abstract supertype for open universe types in "identifier form".

    OUPMType{Int}
Abstract supertype for open universe types in "index form".
"""
abstract type OUPMType{T} end

"""
    oupm_type(::OUPMType)

Get the open universe type for an open universe object.

# Example
```julia
julia> @type AudioSource
AudioSource

julia> oupm_type(AudioSource{Int}(1))
AudioSource
```
"""
function oupm_type(t::OUPMType)
    error("No function `oupm_type` found for $t::$(typeof(t)). Make sure oupm types are declared with @type.")
end

"""
    @type TypeName

Declares the existance of open universe type `TypeName`.
"""
macro type(name::Symbol)
    quote
        struct $(esc(name)){T} <: OUPMType{T}
            idx_or_id::T
            $(esc(name))(x::Int) = new{Int}(x)
            $(esc(name))(x::UUID) = new{UUID}(x)
        end
        $(@__MODULE__).oupm_type(::$(esc(name))) = $(esc(name))

        $(esc(name)) # the "return" from @type should be the type
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