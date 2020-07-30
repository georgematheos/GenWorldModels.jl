"""
    OUPMTypeIdxChange

Diff for an index-form OUPMType which has had its index change value.
"""
struct OUPMTypeIdxChange <: Gen.Diff end

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
oupm_type(t::Diffed{<:OUPMType, NoChange}) = Diffed(oupm_type(strip_diff(t)), NoChange())
oupm_type(t::Diffed{<:OUPMType, OUPMTypeIdxChange}) = Diffed(oupm_type(strip_diff(t)), NoChange())
oupm_type(t::Diffed{<:OUPMType, UnknownChange}) = Diffed(oupm_type(strip_diff(t)), UnknownChange())

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
        $(esc(name))(d::Diffed{Int, Gen.NoChange}) = Gen.Diffed($(esc(name))(Gen.strip_diff(d)), NoChange())
        $(esc(name))(d::Diffed{Int, Gen.UnknownChange}) = Gen.Diffed($(esc(name))(Gen.strip_diff(d)), OUPMTypeIdxChange())
        $(esc(name)) # the "return" from @type should be the type
    end
end

idx(t::OUPMType{Int}) = t.idx_or_id
idx(d::Diffed{<:OUPMType{Int}, UnknownChange}) = Diffed(idx(strip_diff(d)), UnknownChange())
idx(d::Diffed{<:OUPMType{Int}, OUPMTypeIdxChange}) = Diffed(idx(strip_diff(d)), UnknownChange())
idx(d::Diffed{<:OUPMType{Int}, NoChange}) = Diffed(idx(strip_diff(d)), NoChange())

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

export @type
export BirthMove, DeathMove, SplitMove, MergeMove, MoveMove
export UpdateWithOUPMMovesSpec