# TODO: could used a named tuple at the top level; this may improve performance
# could also consider using a regular Dict at the bottom level.
struct ConcreteToAbstractTable
    map::PersistentHashMap{Symbol, PersistentHashMap{Tuple, PersistentHashMap{Int, AbstractOUPMObject}}}
end
@inline function Base.getindex(c::ConcreteToAbstractTable, obj::ConcreteIndexOUPMObject{T}) where {T}
    c.map[T][obj.origin][obj.idx]
end
@inline function Base.haskey(c::ConcreteToAbstractTable, obj::ConcreteIndexOUPMObject{T}) where {T}
    haskey(c.map, T) && haskey(c.map[T], obj.origin) && haskey(c.map[T][obj.origin], obj.idx)
end
function FunctionalCollections.assoc(c::ConcreteToAbstractTable, conc::ConcreteIndexOUPMObject{T}, abst::AbstractOUPMObject{T}) where {T}
    if haskey(c.map, T)
        if haskey(c.map[T], conc.origin)
            m = assoc(c.map, T, assoc(c.map[T], conc.origin, assoc(c.map[T][conc.origin], conc.idx, abst)))
        else
            inner_map = PersistentHashMap{Int, AbstractOUPMObject{T}}()
            inner_map = assoc(inner_map, conc.idx, abst)
            m = assoc(c.map, T, assoc(c.map[T], conc.origin, inner_map))
        end
    else
        deep_inner_map = PersistentHashMap{Int, AbstractOUPMObject{T}}()
        deep_inner_map = assoc(deep_inner_map, conc.idx, abst)
        shallow_inner_map = PersistentHashMap{Tuple, PersistentHashMap{Int, AbstractOUPMObject{T}}}()
        shallow_inner_map = assoc(shallow_inner_map, conc.origin, obj)
        m = assoc(c.map, T, shallow_inner_map)
    end
    ConcreteToAbstractTable(m)
end
function FunctionalCollections.dissoc(c::ConcreteToAbstractTable, conc::ConcreteIndexOUPMObject{T}) where {T}
    top = c.map[T][conc.origin]
    if length(top) == 1
        if !haskey(top, conc.idx)
            throw(keyError(conc))
        else
            m = dissoc(c.map[T], conc.origin)
        end
    else
        m = assoc(t.map[T], conc.origin, dissoc(t.map[T][conc.origin], conc.idx))
    end
    ConcreteToAbstractTable(m)
end

# invariant: the concrete index oupm objects are stored with the origins in FULLY ABSTRACT
# form. so only the index is concrete.
struct IDTable
    concrete_to_abstract::PersistentHashMap{ConcreteIndexOUPMObject, AbstractOUPMObject}
    abstract_to_concrete::PersistentHashMap{AbstractOUPMObject, ConcreteIndexOUPMObject}
end
function IDTable()
    IDTable(
        PersistentHashMap{ConcreteIndexOUPMObject, AbstractOUPMObject}(),
        PersistentHashMap{AbstractOUPMObject, ConcreteIndexOUPMObject}()
    )
end

function Base.show(io::IO, table::IDTable)
    # we should always have the same ids and idxs in the idx_to_id and id_to_idx table,
    # but we retain the capacity to print buggy, nonsymmetric tables to help with debugging
    println("----------------- ID Table -----------------")
    for (concrete, abstract) in table.concrete_to_abstract
        if haskey(table.abstract_to_concrete, abstract) && table.abstract_to_concrete[abstract] == concrete
            println(io, " $concrete <-> $abstract")
        end
    end
    for (abstract, concrete) in table.abstract_to_concrete
        if !haskey(table.concrete_to_abstract, concrete) || table.concrete_to_abstract[concrete] != abstract
            println(io, " $concrete --> $abstract")
        end
    end
    for (concrete, abstract) in table.concrete_to_abstract
        if !haskey(table.abstract_to_concrete, abstract) || table.abstract_to_concrete[abstract] != concrete
            println(io, " $concrete <-- $abstract")
        end
    end
    println("--------------------------------------------")
end

function concrete_origin_error(concrete::ConcreteIndexOUPMObject)
    error("""
        Attempted to perform an ID table operation for $concrete which may only be performed for
        a fully abstract object, or an object with concrete index but fully abstract origin.
    """)
end

get_concrete(table::IDTable, abstract::AbstractOUPMObject) = table.abstract_to_concrete[abstract]
@inline function get_abstract(table::IDTable, concrete::ConcreteIndexAbstractOriginOUPMObject{T}) where {T}
    table.concrete_to_abstract[concrete]
end
get_abstract(table::IDTable, concrete::ConcreteIndexOUPMObject) = concrete_origin_error(concrete)

Base.haskey(table::IDTable, abstract::AbstractOUPMObject) = haskey(table.abstract_to_concrete, abstract)
Base.haskey(table::IDTable, concrete::ConcreteIndexOUPMObject) = haskey(table.concrete_to_abstract, concrete)
Base.getindex(table::IDTable, abs::AbstractOUPMObject) = get_concrete(table, abs)
Base.getindex(table::IDTable, conc::ConcreteIndexOUPMObject) = get_abstract(table, conc)

"""
    assoc(table, concrete, abstract)
    assoc(table, abstract, concrete)

Returns an updated version of `table` where the `concrete` object (with origin fully abstract)
with the given `abstract` object, potentially overwriting any current
associations for `concrete` and `abstract`.
"""
function FunctionalCollections.assoc(table::IDTable, abstract::AbstractOUPMObject{T}, concrete::ConcreteIndexOUPMObject{T, OT}) where
            {T, OT <: Tuple{Vararg{<:AbstractOUPMObject}}}
    new_atoc = assoc(table.abstract_to_concrete, abstract, concrete)
    new_ctoa = assoc(table.concrete_to_abstract, concrete, abstract)
    IDTable(new_ctoa, new_atoc)
end
FunctionalCollections.assoc(::IDTable, ::AbstractOUPMObject{T}, c::ConcreteIndexOUPMObject{T}) where {T} = concrete_origin_error(c)
FunctionalCollections.assoc(t::IDTable, c::ConcreteIndexOUPMObject, a::AbstractOUPMObject) = assoc(t, a, c)

"""
    generate_abstract_for(id_table, concrete_obj)

Generate an abstract form for the given concrete object (which has origin
in fully abstract form), and return an updated version of `id_table` with this
abstract form associated with the concrete object.
"""
function generate_abstract_for(table::IDTable, concrete::ConcreteIndexAbstractOriginOUPMObject{T}) where {T}
    id = gensym()
    abstract = AbstractOUPMObject{T}(id)
    return (assoc(table, concrete, abstract), abstract)
end

"""
    move_all_between(table::IDTable, typename, origin; min, max=Inf, inc)

Update the abstract<-->concrete associations for objects of type 
`OUPMObject{typename}` in the ID table with the given `origin` as follows.
For every abstract object whose associated index is `idx` within the range `min<=...<=max`,
change the association so the abstract object is associated with index `idx + inc`.
Delete the association for the abstract objects whose concrete objects' associations
have been overwritten, and delete the associaiton for concrete objects whose abstract objects'
associations have been overwritten.

Return `(new_table, changed_abstract_objects)` where `new_table` is the updated table,
and `changed_abstract_objects` is a set of all the abstract objects which have been deleted or moved.

This update may result in some indices which previously had associated identifiers
now having no associated identifier.
The update may also overwrite current associations for indices in the range
(min+inc,...min) or (max...max+inc).
"""
function move_all_between(table::IDTable, typename::Symbol, origin::Tuple{Vararg{<:AbstractOUPMObject}}; min=1, inc, max=Inf)
    original_c_to_a = table.concrete_to_abstract
    cobj(idx) = ConcreteIndexOUPMObject{typename}(origin, idx)
    new_c_to_a = table.concrete_to_abstract
    new_a_to_c = table.concrete_to_abstract

    changed_abstract_objs = Set{AbstractOUPMObject}()
    # TODO: should I return information about the deleted objects seperately?

    # at each step through this loop, we will update
    # id_to_idx_for_type[id].
    # if we are moving this ID, we will also update idx_to_id_for_type[idx+inc].
    # if there is no ID which will be moved to position `idx`, we also
    # update idx_to_id_for_type[idx]
    for (concrete, abstract) in original_c_to_a[typename][origin]
        idx = concrete.idx

        if (min <= idx && idx <= max)
            # all ids in the min<=...<=max range have their index incremented by `inc`
            new_c_to_a = assoc(new_c_to_a, abstract, cobj(idx+inc))
            new_a_to_c = assoc(new_a_to_c, cobj(idx+inc), abstract)
            push!(changed_abstract_objs, abstract)
        elseif (min+inc <= idx && idx < min) || (max < idx && idx <= max+inc)
            # all ids with current index in an area IDs are getting moved to, but where
            # ids are not moving anywhere, should be removed since they are going to be overwritten
            new_a_to_c = dissoc(new_a_to_c, abstract)
            push!(changed_abstract_objs, abstract)
        end

        if (min+inc <= idx && idx <= max+inc)
            # all indices in the range where IDs are being moved will either
            # 1. have an ID moved to this index, in which case the above `if` statement handles the id change
            # or
            # 2. have no ID which is going to get moved to it, and hence needs to have its current id deleted
            # we handle (2) here.
            if !haskey(original_c_to_a, idx-inc)
                new_c_to_a = dissoc(new_c_to_a, cobj(idx))
            end
        elseif (min <= idx && idx < min+inc) || (max+inc < idx && idx <= max)
            # all indices in an area where IDs are moving from, but no IDs are moving to,
            # should be removed
            new_c_to_a = dissoc(new_c_to_a, cobj(idx))
        end
    end
    new_table = IDTable(new_c_to_a, new_a_to_c)
    return (new_table, changed_abstract_objs)
end