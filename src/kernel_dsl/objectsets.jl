using Memoization: @memoize

# we need this since we can't use the `_get_sibling_set` generative
# function outside the model
struct SibSetSpec{O} <: ObjectSetSpec
    typename::Symbol
    origin::O
end

function _construct_object_set(tr, spec::SingletonObjectSetSpec)
    return Set([spec.obj])
end
function _construct_object_set(tr, spec::TypeObjectSetSpec)
    origin_sig_table = get_gen_fn(tr).meta.origin_sig_table
    sigs = origin_sig_table[spec.typename]
    return union((_get_object_set(tr, get_origin_sig_spec(s)) for s in sigs)...)
end
function _construct_object_set(tr, spec::OriginConstrainedObjectSetSpec)
    origin_sets = (_get_object_set(tr, constraint) for constraint in spec.constraints)
    all_origins = Iterators.product(origin_sets)
    return union((
        _get_object_set(tr, SibSetSpec(spec.typename, origin)) for origin in all_origins
    )...)
end
function _construct_object_set(tr, spec::SibSetSpec)
    num_stmt_name = num_statement_name(OriginSignature(
        spec.typename,
        Tuple(typeof(obj) for obj in spec.origin)
    ))
    num = tr[:world => num_stmt_name => spec.origin]
    return Set((OUPMObject{spec.typename}(i) for i=1:num))
end

# to get an object set, check if we already constructed it while running the model,
# otherwise, construct it here (using a memoization cache to turn this into DP)
@memoize function _get_object_set(tr, spec::ObjectSetSpec)
    if !isa(spec, SibSetSpec) && has_val(tr.world, Call(:_objects, spec))
        get_val(tr.world, Call(:_objects, spec))
    else
        _construct_object_set(tr, spec)
    end
end

macro objects(tr, spec_expr)
    objset_spec = get_spec(spec_expr, __module__)
    :(_get_object_set($(tr, objset_spec)))
end