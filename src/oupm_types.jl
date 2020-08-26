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

function concrete_index_oupm_object(name, origin, idx)
    ConcreteIndexOUPMObject{name}(origin, idx)
end
function concrete_index_oupm_object(name::Diffed{Symbol}, origin::Diffed, idx)
    Diffed(concrete_index_oupm_object(strip_diff(name), strip_diff(origin), strip_diff(idx)), UnknownChange())
end
function concrete_index_oupm_object(name::Diffed{Symbol, NoChange}, origin::Diffed{<:Any, NoChange}, idx::Diffed{<:Any, NoChange})
    Diffed(concrete_index_oupm_object(strip_diff(name), strip_diff(origin), strip_diff(idx)), NoChange())
end

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
oupm_type_name(::OUPMObject{T}) where {T} = T
oupm_type_name(::Diffed{<:OUPMObject{T}, NoChange}) where {T} = Diffed(T, NoChange())
oupm_type_name(::Diffed{<:OUPMObject{T}, UnknownChange}) where {T} = Diffed(T, UnknownChange())

Base.getproperty(o::Diffed{<:ConcreteIndexOUPMObject, NoChange}, sym::Symbol) = Diffed(Base.getproperty(strip_diff(o), sym), NoChange())
Base.getproperty(o::Diffed{<:ConcreteIndexOUPMObject, UnknownChange}, sym::Symbol) = Diffed(Base.getproperty(strip_diff(o), sym), UnknownChange())

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

function concrete_origin_error(concrete::ConcreteIndexOUPMObject)
    @assert !(concrete isa ConcreteIndexAbstractOriginOUPMObject) "This error should never be thrown for an object with fully abstract origin!"
    error("""
        Attempted to perform an operation for $concrete which may only be performed for
        a fully abstract object, or an object with concrete index but fully abstract origin.
    """)
end

Base.isapprox(x::OUPMObject, y::OUPMObject) = isequal(x, y)

export @type
export OUPMObject, AbstractOUPMObject, ConcreteIndexOUPMObject, ConcreteIndexAbstractOriginOUPMObject