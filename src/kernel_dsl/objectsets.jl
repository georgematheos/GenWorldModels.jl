using Memoization: @memoize

# we need this since we can't use the `_get_sibling_set` generative
# function outside the model
struct SibSetSpec{O} <: ObjectSetSpec
    typename::Symbol
    origin::O
end

function _construct_object_set(tr, spec::SingletonObjectSetSpec)
    println("SingSet $(spec.typename)")
    return Set([spec.obj])
end
function _construct_object_set(tr, spec::TypeObjectSetSpec)
    println("TypeSet $(spec.typename)")
    origin_sig_table = (Gen.get_gen_fn(tr)).meta.origin_sig_table
    sigs = origin_sig_table[spec.typename]
    union((_get_object_set(tr, get_origin_sig_spec(s)) for s in sigs)...)
end
function _construct_object_set(tr, spec::OriginConstrainedObjectSetSpec)
    println("OgSet $(spec.typename)")
    origin_sets = (_get_object_set(tr, constraint) for constraint in spec.constraints)
    all_origins = Iterators.product(origin_sets...)
    println("all_origins: ", collect(all_origins))
    return union((
        _get_object_set(tr, SibSetSpec(spec.typename, origin)) for origin in all_origins
    )...)
end
function _construct_object_set(tr, spec::SibSetSpec)
    println("SibSet $(spec.typename)")
    @assert (spec.typename isa Symbol) "spec: $spec"
    num_stmt_name = num_statement_name(OriginSignature(
        spec.typename,
        Tuple(oupm_type_name(obj) for obj in spec.origin)
    ))
    num = tr[:world => num_stmt_name => spec.origin]
    println("num: $num")
    println("obj origin: $(spec.origin)")
    return Set((OUPMObject{spec.typename}(spec.origin, i) for i=1:num))
end

@memoize function _get_object_set(tr, spec::ObjectSetSpec)
    _construct_object_set(tr, spec)
end

# TODO: docstring
macro objects(tr, spec_expr)
    objset_spec = get_spec(spec_expr, __module__)
    :(_get_object_set($(esc(tr)), $(objset_spec)))
end