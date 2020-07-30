"""
    OUPMObject{T}

An open universe object whose object type has name given by the symbol `T`.
"""
abstract type OUPMObject{T} end
struct AbstractOUPMObject{T} <: OUPMObject{T}
    id::Symbol
end
struct ConcreteIndexOUPMObject{T, OriginType} <: OUPMObject{T}
    origin::OriginType
    idx::Int
    function ConcreteIndexOUPMObject{T}(origin::OT, idx::Int) where {T, OT <: Tuple{Vararg{<:OUPMObject}}}
        new{T, OT}(origin, idx)
    end
end
(OUPMObject{T})(origin::Tuple, idx::Int) where {T} = ConcreteIndexOUPMObject{T}(origin, idx)
(OUPMObject{T})(idx::Int) where {T} = ConcreteIndexOUPMObject{T}((), idx)
function ConcreteIndexOUPMObject{T}(origin::OT, idx::Int) where {T, OT}
    error("Tried to create a $T with origin $origin which is not a tuple of OUPMObjects.")
end
const ConcreteIndexAbstractOriginOUPMObject{T} = ConcreteIndexOUPMObject{T, <:Tuple{Vararg{<:AbstractOUPMObject}}}

function Base.show(io::IO, obj::ConcreteIndexOUPMObject{T}) where {T}
    print(io, T)
    print(io, "(")
    if obj.origin !== ()
        print(io, obj.origin)
        print(io, ", ")
    end
    print(io, obj.idx)
    print(io, ")")
end
function Base.show(io::IO, obj::AbstractOUPMObject{T}) where {T}
    print(io, T)
    print(io, "(")
    print(io, obj.id)
    print(io, ")")
end
typename(::OUPMObject{T}) where {T} = T

"""
    @type TypeName

Declares the existance of open universe type `TypeName`.
"""
macro type(name::Symbol)
    quote
        $(esc(name)) = OUPMObject{$(QuoteNode(name))}
        $(esc(name)) # the "return" from @type should be the type
    end
end


# TODO: below here

# abstract type OUPMMove end
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
# struct MoveMove <: OUPMMove
#     type::Type{<:OUPMType}
#     from_idx::Int
#     to_idx::Int
# end

# struct UpdateWithOUPMMovesSpec <: Gen.CustomUpdateSpec
#     moves::Tuple{Vararg{<:OUPMMove}}
#     subspec::Gen.UpdateSpec
# end

export @type
export OUPMObject, AbstractOUPMObject, ConcreteIndexOUPMObject, ConcreteIndexAbstractOriginOUPMObject
# export BirthMove, DeathMove, SplitMove, MergeMove, MoveMove
# export UpdateWithOUPMMovesSpec