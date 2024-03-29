#=
Generative functions to construct object sets according to number statements.

Examples of the `@objects` command:
- `@objects City` (all cities)
- `@objects City(california)` (all cities in California)
- `@objects City(State)` (all cities associated with any state)
- `@objects City(State(usa))` (all cities in a state in the USA)
- `@objects City(State(usa), california)` (all cities in 2 states, the first of which is in the USA, and the second of which is California)

The general syntax is
- `@objects spec`

where `spec` = object | typename | typename(Vararg(spec)).
- `@objects object` = Set([object])
- `@objects typename` = all objects of the given type
- `@objects typename(spec1, spec2, ..., specn)` = all objects of the given type whose
    first origin object matches spec1, second origin object matches spec2, ..., nth origin object matches specn
=#

### types of object specs ###
abstract type ObjectSetSpec end
struct SingletonObjectSetSpec <: ObjectSetSpec
    obj::OUPMObject
end
struct TypeObjectSetSpec <: ObjectSetSpec
    typename::Symbol
end
struct OriginConstrainedObjectSetSpec <: ObjectSetSpec
    typename::Symbol
    constraints::Tuple{Vararg{ObjectSetSpec}}
end

### @objects macro ###
"""
    @objects object_set_spec_expr

Used in the OUPM DSL, this gets the set of all objects satisfying the given `object_set_spec_expr`.
Valid object set specifications have one of the following forms:
- `@objects TypeName` gets all objects of the given type
- `@objects object` gets the set of the single object bound to the variable `object`
- `@objects Typename(spec1, ..., specn)` gets the set of all objects of the given type with an origin
   `(s_1, ..., s_n)`, where `s_j` satisfies `spec_j` for every j.+

"""
macro _objects(world, expr)
    :(lookup_or_generate($(esc(world))[:_objects][$(get_spec(expr, __module__))]))
end

# get the ObjectSetSpec corresponding to the given expression or symbol
function get_spec(name::Symbol, __module__)
    if isdefined(__module__, name)
        val = Base.eval(__module__, name)
        if val isa Type && val <: OUPMObject
            return :(TypeObjectSetSpec($(QuoteNode(name))))
        end
    end
    return :(SingletonObjectSetSpec($(esc(name))))
end
get_spec(expr::Expr, __module__) =
    if MacroTools.@capture(expr, name_(subspecs__))
        if !isempty(subspecs) && subspecs[end] isa Integer
            # in this case, this is a constructor for a concrete object, eg.
            # Airplane(1) or Child(Parent(1), 3)
            :(SingletonObjectSetSpec($(esc(expr))))
        else
            :(
                OriginConstrainedObjectSetSpec(
                    $(QuoteNode(name)),
                    $(Expr(:tuple, (get_spec(subspec, __module__) for subspec in subspecs)...))
                )
            )
        end
    else
        error("Unrecognized @objects spec: $expr")
    end


### getters ###
# `_objects` should be a MGF in every world using this, with address :_objects

"""
    get_object_set_getter_expressions(origin_sig_table_name)

Returns an expression defining the _objects generative function, given
the symbol to which a type-to-originsignature table is bound for the given world model.
(This table specifies, for each type, what types of origins objects of that type can have,
ie. which origin tuples number statements have been provided for.)
Returns `(expr, fn_name)` where `expr` is the function-defining expression, and `fn_name`
is the name of the defined function.

This will define 2 generative functions: _type_objects (to get object sets for `TypeObjectSetSpec`s),
and a function with name `gensym("_objects")` which performs dispatch to specific set getters
for each type of object set spec.
"""
function get_object_set_getter_expressions(origin_sig_table_name)
    fn_name = gensym("_objects")
    expr = quote
        @gen (static, diffs) function _type_objects(world, spec::$(@__MODULE__).TypeObjectSetSpec)
            sigs = $origin_sig_table_name[spec.typename]
            sig_to_spec = lazy_set_to_dict_map($(@__MODULE__).get_origin_sig_spec, sigs)
            sig_to_set ~ dictmap_lookup_or_generate(world[:_objects], sig_to_spec)
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
    return Set([spec.obj])
end

function get_origin_sig_spec(sig)
    OriginConstrainedObjectSetSpec(
        sig.typename,
        Tuple(TypeObjectSetSpec(origintypename) for origintypename in sig.origin_typenames)
    )
end

function num_statement_name(spec::OriginConstrainedObjectSetSpec)
    origin_names = Tuple(
        if c isa TypeObjectSetSpec || c isa OriginConstrainedObjectSetSpec
            c.typename
        else
            oupm_type_name(c.obj)
        end
        for c in spec.constraints
    )
    num_statement_name(OriginSignature(spec.typename, origin_names))
end
num_statement_name(s::Diffed{<:Any, NoChange}) = Diffed(num_statement_name(strip_diff(s)), NoChange())
num_statement_name(s::Diffed{<:Any}) = Diffed(num_statement_name(strip_diff(s)), UnknownChange())

@gen (static, diffs) function _constrained_objects(world, spec::OriginConstrainedObjectSetSpec)
    origin_sets ~ map_lookup_or_generate(world[:_objects], spec.constraints)
    
    all_origins ~ tracked_product_set(origin_sets)
    origins_to_specs = GetOriginsToSiblingSetSpecs(spec.typename, num_statement_name(spec))(world, all_origins)
    origins_to_sets ~ DictMap(_get_sibling_set)(origins_to_specs)
    set ~ tracked_union(origins_to_sets)

    return set
end

### construct a table of origin signatures per type ###
function add_origin_sig_table!(stmts, meta)
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

    return originsig_name
end

### add objectset getters to an OUPM DSL expansion ###
"""
    add_objectset_getters_to_expansion!(stmts, meta)

Adds expressions to `stmts` needed for object set getting (`@objects`)
syntax.  Returns the tuple (addr, fnname),
where `fnname` is the name of a generative function for object set getting,
which should be memoized at address `addr` in the `UsingWorld` combinato.
"""
function add_objectset_getters_to_expansion!(stmts, origin_sig_table_name)
    (expr, fn_name) = get_object_set_getter_expressions(origin_sig_table_name)
    push!(stmts, expr)
    return (:_objects, fn_name)
end