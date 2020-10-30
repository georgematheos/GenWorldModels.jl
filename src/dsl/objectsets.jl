#=
Generative functions to construct object sets according to number statements.

Examples of the `@objects` command:
- `@objects City` (all cities)
- `@objects City(california)` (all cities in California)
- `@objects City(State)` (all cities associated with any state)
- `@objects City(State(usa))` (all cities in a state in the USA)
- `@objects City(State(usa), State(usa))` (all cities in 2 states, both in the USA)

The general syntax is
- `@objects spec`

where `spec` = object | typename | typename(Vararg(spec)).
- `@objects object` = Set([object])
- `@objects typename` = all objects of the given type
- `@objects typename(spec1, spec2, ..., specn)` = all objects of the given type whose
    first origin object matches spec1, second origin object matches spec2, ..., nth origin object matches specn
=#

abstract type ObjectSetSpec end
struct SingletonObjectSetSpec <: ObjectSetSpec
    obj::OUPMObject
end
struct TypeObjectSetSpec <: ObjectSetSpec
    typename::Symbol
end
struct OriginConstrainedObjectSetSepc <: ObjectSetSpec
    typename::Symbol
    constraints::Tuple{Vararg{ObjectSetSpec}}
end

### @objects macro ###
macro objects(world, expr)
    :(lookup_or_generate($(esc(world))[:_objects][$(get_spec(expr, __module__))]))
end
function get_spec(name::Symbol, __module__)
    if isdefined(__module__, name) && Base.eval(__module__, name) <: OUPMObject
        expr = :(TypeObjectSetSpec($(QuoteNode(name))))
    else
        expr = :(SingletonObjectSetSpec($(esc(name))))
    end

    expr
end
function get_spec(expr::Expr, __module__)
    if MacroTools.@capture(expr, name_(subspecs__))
        :(
            OriginConstrainedObjectSetSepc(
                $name,
                $(Expr(:tuple, (get_spec(subspec, __module__) for subspec in subspecs)...))
            )
        )
    else
        error("Unrecognized @objects spec: $expr")
    end
end

### getters ###
# `_objects` should be a MGF in every world using this, with address :_objects

function get_object_set_getter_expressions(origin_sig_table_name)
    fn_name = gensym("_objects")
    expr = quote
        @gen (static, diffs) function _type_objects(world, spec::$(@__MODULE__).TypeObjectSetSpec)
            sigs = $origin_sig_table_name[spec.typename]
            sig_to_spec = lazy_set_to_dict_map($(@__MODULE__).get_origin_sig_spec, sigs)
            sig_to_set ~ DictMap(lookup_or_generate)(mgfcall_dictmap(world[:_objects], sig_to_spec))
            set ~ tracked_union(sig_to_set)

            return set
        end

        @gen (static, diffs) function $fn_name(world, spec::$(@__MODULE__).ObjectSetSpec)
            branch = switchint(spec isa $(@__MODULE__).SingletonObjectSetSpec, spec isa $(@__MODULE__).TypeObjectSetSpec)
            set ~ Switch($(@__MODULE__)._singleton_objects, _type_objects, $(@__MODULE__)._constrained_objects)(branch, world, spec)
            return set
        end
    end
    return (expr, fn_name)
end

@gen (static, diffs) function _singleton_objects(world, spec::SingletonObjectSetSpec)
    return Set([spec.object])
end

function get_origin_sig_spec(sig)
    OriginConstrainedObjectSetSepc(
        sig.typename,
        Tuple(TypeObjectSetSpec(origintypename) for origintypename in sig.origin_typenames)
    )
end

function num_statement_name(spec::OriginConstrainedObjectSetSepc)
    origin_names = Tuple(
        if c isa TypeObjectSetSpec || c isa OriginConstrainedObjectSetSepc
            c.typename
        else
            oupm_type_name(c.obj)
        end
        for c in spec.constraints
    )
    num_statement_name(OriginSignature(spec.typename, origin_names))
end
@gen (static, diffs) function _constrained_objects(world, spec::OriginConstrainedObjectSetSepc)
    origin_sets ~ Map(lookup_or_generate)(mgfcall_map(world[:_objects], spec.constraints))
    
    all_origins ~ tracked_product_set(origin_sets)
    origins_to_specs = GetOriginsToSiblingSetSpecs(spec.typename, num_statement_name(spec))(world, all_origins)
    origins_to_sets ~ DictMap(_get_sibling_set)(origins_to_specs)
    set ~ tracked_union(origins_to_sets)

    return set
end

### add objectset getters to an OUPM DSL expansion ###
function add_objectset_getters_to_expansion!(stmts, meta)
    originsig_name = gensym("origin_signatures")

    type_to_sig = Dict()
    for sig in keys(meta.number_stmts)
        if !(sig.typename in keys(type_to_sig))
            type_to_sig[sig.typename] = [sig]
        else
            push!(type_to_sig[sig.typename], sig)
        end
    end

    sig_expr(sig) = :($(GlobalRef(@__MODULE__, :OriginSignature))(
        $(QuoteNode(sig.typename)),
        $(Expr(:tuple, (QuoteNode(name) for name in sig.origin_typenames)...))
    ))
    sigs_expr(sigs) = :(Set([ $(map(sig_expr, sigs)...) ]))
    originsig_list_expr = :(
        Dict($((
            :($(QuoteNode(typename)) => $(sigs_expr(sigs)))
            for (typename, sigs) in type_to_sig
        )...))
    )

    push!(stmts, :($originsig_name = $originsig_list_expr))
    (expr, fn_name) = get_object_set_getter_expressions(originsig_name)
    push!(stmts, expr)
    return (:_objects, fn_name)
end