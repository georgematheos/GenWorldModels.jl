#=
Commands for interacting with the world in the OUPM DSL.

"Root" Commands:
- `@get(world, propertyname[obj1, obj2, ..., objn])`
- `@arg(world, argname)`
- `@origin(world, obj)` or `@index(world, obj)`

These will all be expanded to into a `lookup_or_generate(world[...][...])`, which
should be traced.

These may be mapped:
- `@map({...} ~ $(GlobalRef(GenWorldModels, lookup_or_generate))(world[...][...]) for ... in ...)`
- Likewise for `@setmap` and `@nocollision_setmap`

The idea is that macroexpansion should occur via a depth-first traversal, which will first convert
`@get`, `@origin`, and `@index` commands to `{...} ~ GenWorldModels.lookup_or_generate(world[...][...])`,
and then an outer `@map`, `@setmap`, or `@nocollision_setmap` can transform a comprehension containing this
into the proper `Map(lookup_or_generate)...`, etc., command.

Examples of valid map expressions:
```
reading_list = @map [@get(reading[detector]) for detector in detector_sample_list]
reading_set = @setmap [@get(reading[detector]) for detector in detector_set]
origins = @map (@origin(obj) for obj in objects)
is_parent_matrix = @map [@get(is_parent[obj1, obj2]) for (obj1, obj2) in collect(Iterators.product(objects_list, objects_list))]
```
=#
function _map(log_expr, MapCombinator, arg_mapper, lazymapper)
    log = GlobalRef(@__MODULE__, :lookup_or_generate)
    if ( # for some reason, using the macrotools | union is not working here.
        MacroTools.@capture(log_expr, [addr_ ~ $log(world_[name_][key_]) for item_ in list_]) ||
        MacroTools.@capture(log_expr, (addr_ ~ $log(world_[name_][key_]) for item_ in list_))
    )
        # note that we throw away `addr`; a new one will be generated.
        # also note that we don't put `name` in a QuoteNode, since the @get or @origin or @index (etc.)
        # will have already done this
        if name == item
            :($(MapCombinator)(lookup_or_generate)($(arg_mapper)($(esc(world))[$name], $(esc(list)))))
        else
            :($(MapCombinator)(lookup_or_generate)($(arg_mapper)($(esc(world))[$name], $(lazymapper)($item -> $key , $(esc(list))))))
        end
    else
        error("Unexpected @map expression after partial parsing: $log_expr")
    end
end

# the first arg will be the world's name, which we don't use, since it should be in the
# already-parsed lookup_or_generate
macro map(_, log_expr)
    _map(log_expr, :Map, :mgfcall_map, :lazy_map)
end
macro setmap(_, log_expr)
    _map(log_expr, :SetMap, :mgfcall_setmap, :lazy_no_collision_set_map)
end
macro nocollision_setmap(_, log_expr)
    _map(log_expr, :NoCollisionSetMap, :mgfcall_setmap, :lazy_no_collision_set_map)
end
# TODO: dictmap

macro origin(world, obj)
    :(lookup_or_generate($(esc(world))[:origin][$(esc(obj))]))
end

macro index(world, obj)
    :(lookup_or_generate($(esc(world))[:index][$(esc(obj))]))
end

# TODO: should we include @concrete & @abstract?

macro get(world, expr::Expr)
    @assert expr.head === :ref "Invalid get statement: @get $expr"
    propname = expr.args[1]
    key = Expr(:tuple, (esc(a) for a in expr.args[2:end])...)
    :(lookup_or_generate($(esc(world))[$(QuoteNode(propname))][$key]))
end

macro arg(world, argname)
    :(lookup_or_generate($(esc(world))[:args][$(QuoteNode(argname))]))
end

const DSL_COMMANDS = [
    Symbol("@$name") for name in
    (:origin, :index, :arg, :get, :map, :setmap, :nocollision_setmap, :objects)
]

# Question: do we want to export some of these?
# Do we want to provide versions of `@origin` and `@index` for concrete objects
# which do not need world access?