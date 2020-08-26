Base.:(:)(fst::Int, lst::Diffed{Int, UnknownChange}) = Diffed(fst:strip_diff(lst), UnknownChange())
concrete_obj_getter(typename, origin) = i -> concrete_index_oupm_object(typename, origin, i)
function concrete_obj_getter(typename::Diffed{Symbol}, origin::Diffed) 
    Diffed(concrete_obj_getter(strip_diff(typename), strip_diff(origin)), UnknownChange())
end
function concrete_obj_getter(typename::Diffed{Symbol, NoChange}, origin::Diffed{<:Any, NoChange}) 
    Diffed(concrete_obj_getter(strip_diff(typename), strip_diff(origin)), NoChange())
end

@gen (static, diffs) function get_sibling_set(spec::SiblingSetSpec)
    num ~ lookup_or_generate(spec.world[spec.num_address][spec.origin])
    concrete_objs ~ no_collision_set_map(concrete_obj_getter(spec.typename, spec.origin), 1:num)
    abstract_objs ~ NoCollisionSetMap(lookup_or_generate)(mgfcall_setmap(spec.world[_get_abstract_addr], concrete_objs))
    return abstract_objs
end
@load_generated_functions()