"""
    generate_abstract_object!(world, concrete_object)

Generate an abstract object for the given `concrete_object`.
Returns the generated abstract object.

All calls to generating an abstract object should pass through this method.
"""
function generate_abstract_object!(world::World, obj::ConcreteIndexAbstractOriginOUPMObject)
    (world.id_table, abstract) = generate_abstract_for(world.id_table, obj)
    note_new_call!(world, Call(_get_index_addr, abstract))
    note_new_call!(world, Call(_get_origin_addr, abstract))
    note_new_call!(world, Call(_get_concrete_addr, abstract))
    note_new_call!(world, Call(_get_abstract_addr, obj))
    return abstract
end

"""
    concrete_to_abstract(world, item)

If `item` is a concrete OUPM object, returns the
abstract form for the item in the world; otherwise simply returns
`item` unchanged.
"""
convert_to_abstract(_, item) = item
convert_to_abstract(world, item::ConcreteIndexAbstractOriginOUPMObject) = get_val(world, Call(_get_abstract_addr, item))
function convert_to_abstract(world, item::ConcreteIndexOUPMObject{T}) where {T}
    conv_to_abst(x) = convert_to_abstract(world, x)
    new_origin = map(conv_to_abst, item.origin)
    return convert_to_abstract(world, ConcreteIndexOUPMObject{T}(new_origin, item.idx))
end

"""
    convert_to_abstract!(world, item)

If `item` is a concrete OUPM object, returns the
abstract form for the item in the world, generating one if none currently exists;
otherwise simply returns `item` unchanged.
"""
convert_to_abstract!(_, item) = item
function convert_to_abstract!(world, item::ConcreteIndexAbstractOriginOUPMObject)
    call = Call(_get_abstract_addr, item)
    if has_val(world, call)
        return get_val(world, call)
    else
        return generate_abstract_object!(world, item)
    end
end
function convert_to_abstract!(world, item::ConcreteIndexOUPMObject{T}) where {T}
    conv_to_abst!(x) = convert_to_abstract!(world, x)
    new_origin = map(conv_to_abst!, item.origin)
    return convert_to_abstract!(world, ConcreteIndexOUPMObject{T}(new_origin, item.idx))
end

"""
    convert_to_concrete_deep(world, item)

If `item` is an abstract OUPM object, returns the
concrete form with fully concrete origin for the item in the world; otherwise simply returns
`item` unchanged.
"""
convert_to_concrete(_, item) = item
function convert_to_concrete(world, item::AbstractOUPMObject{T}) where {T}
    obj = get_val(world, Call(_get_concrete_addr, item))
    to_conc(x) = convert_to_concrete(world, x)
    concrete_origin = map(to_conc, obj.origin)
    return ConcreteIndexOUPMObject{T}(concrete_origin, obj.idx)
end