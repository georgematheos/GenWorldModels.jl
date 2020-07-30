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

# TODO: all origin-->abstract origin should occur in `lookup_or_generate`!

# TODO: below this

# @inline convert_key_to_id_form(world, key) = key
# @inline function convert_key_to_id_form(world, key::OUPMType{Int})
#     type = oupm_type(key)
#     id = get_id(world.id_table, type, key.idx_or_id)
#     type(id) 
# end

# @inline convert_key_to_id_form!(world, key) = key
# @inline function convert_key_to_id_form!(world, key::OUPMType{Int})
#     type = oupm_type(key)
#     idx = key.idx_or_id
#     if has_idx(world.id_table, type, idx)
#         id = get_id(world.id_table, type, idx)
#     else
#         id = generate_id_for_call!(world, Call(type, idx))
#     end
#     return type(id)
# end

# @inline convert_key_to_idx_form(world, key) = key
# @inline function convert_key_to_idx_form(world, key::OUPMType{UUID})
#     type = oupm_type(key)
#     idx = get_idx(world.id_table, type, key.idx_or_id)
#     type(idx)
# end