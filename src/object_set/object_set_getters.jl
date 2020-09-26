##########################
# diff propagation utils #
##########################

Base.:(:)(fst::Int, lst::Diffed{Int, UnknownChange}) = Diffed(fst:strip_diff(lst), UnknownChange())
concrete_obj_getter(typename, origin) = i -> concrete_index_oupm_object(typename, origin, i)
function concrete_obj_getter(typename::Diffed{Symbol}, origin::Diffed) 
    Diffed(concrete_obj_getter(strip_diff(typename), strip_diff(origin)), UnknownChange())
end
function concrete_obj_getter(typename::Diffed{Symbol, NoChange}, origin::Diffed{<:Any, NoChange}) 
    Diffed(concrete_obj_getter(strip_diff(typename), strip_diff(origin)), NoChange())
end

constlen_vec(vals...) = Base.vect(vals...)
function constlen_vec(vals::Vararg{<:Diffed})
    vls = map(strip_diff, vals)
    dfs = map(get_diff, vals)
    df = VectorDiff(length(vls), length(vls), Dict(i => d for (i,d) in enumerate(dfs) if d !== NoChange()))
    Diffed(Base.vect(vls...), df)
end

#################################################
# generative functions for creating object sets #
#################################################

@gen (static, diffs) function get_sibling_set_from_num(typename::Symbol, world::World, origin::Tuple, num::Int)
    concrete_objs ~ no_collision_set_map(concrete_obj_getter(typename, origin), 1:num)
    abstract_objs ~ NoCollisionSetMap(lookup_or_generate)(mgfcall_setmap(world[_get_abstract_addr], concrete_objs))
    return abstract_objs
end
@gen (static, diffs) function _get_sibling_set(spec::SiblingSetSpec)
    num ~ lookup_or_generate(spec.world[spec.num_address][spec.origin])
    objs ~ get_sibling_set_from_num(spec.typename, spec.world, spec.origin, num)
    return objs
end
@gen (static, diffs) function get_sibling_set(typename::Symbol, num_address::Symbol, world::World, origin::Tuple)
    spec = GetSiblingSetSpec(typename, num_address)(world, origin)
    objs ~ _get_sibling_set(spec)
    return objs
end

@gen (static, diffs) function get_origin_iterated_set(typename, num_address, world, origin_sets)
    all_origins ~ tracked_product_set(origin_sets)
    origins_to_specs = GetOriginsToSiblingSetSpecs(typename, num_address)(world, all_origins)
    origins_to_sets ~ DictMap(_get_sibling_set)(origins_to_specs)
    sets ~ tracked_union(origins_to_sets)

    return sets
end
# @load_generated_functions()