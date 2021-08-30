#=
Commands for interacting with the world in the OUPM DSL.

"Core" Commands:
- `@get(world, propertyname[obj1, obj2, ..., objn])`
- `@arg(world, argname)`
- `@origin(world, obj)` or `@index(world, obj)`

These will all be expanded to into a `lookup_or_generate(world[...][...])`, which
should be traced.

These may be mapped:
- `@map({...} ~ $(GlobalRef(GenWorldModels, lookup_or_generate))(world[...][...]) for ... in ...)`
- Likewise for `@nocollision_setmap`
- For `@dictmap`, the syntax should be `@dictmap({} ~ ...)(k => world[...][...] for (k, ...) in ...)`

The idea is that macroexpansion should occur via a depth-first traversal, which will first convert
`@get`, `@origin`, and `@index` commands to `{...} ~ GenWorldModels.lookup_or_generate(world[...][...])`,
and then an outer `@map`, `@dictmap`, or `@nocollision_setmap` can transform a comprehension containing this
into the proper `map_lookup_or_generate`, etc., command.

Examples of valid map expressions:
```
reading_list = @map [@get(reading[detector]) for detector in detector_sample_list]
reading_set = @dictmap [d => @get(reading[detector]) for (d, detector) in lazy_set_to_dict_map(identity, detector_set])
origins = @map (@origin(obj) for obj in objects)
is_parent_matrix = @map [@get(is_parent[obj1, obj2]) for (obj1, obj2) in collect(Iterators.product(objects_list, objects_list))]
@dictmap (k => @get(property[foo]) for (k, foo) in dict)
```
=#
function _map(log_expr, mapped_lookup_or_generate, lazymap)
    log = GlobalRef(@__MODULE__, :lookup_or_generate)
    if ( # for some reason, using the macrotools | union is not working here.
        MacroTools.@capture(log_expr, [addr_ ~ $log(world_[name_][key_]) for item_ in list_]) ||
        MacroTools.@capture(log_expr, (addr_ ~ $log(world_[name_][key_]) for item_ in list_))
    )
        # note that we throw away `addr`; a new one will be generated.
        # also note that we don't put `name` in a QuoteNode, since the @get or @origin or @index (etc.)
        # will have already done this
        if name == item
            :($(mapped_lookup_or_generate)($(esc(world))[$name], $(esc(list))))
        else
            :($(mapped_lookup_or_generate)($(esc(world))[$name], $(lazymap(item, key, list))))
        end
    else
        error("Unexpected @map/@nocollision_setmap expression after partial parsing: $log_expr")
    end
end

function map_transform(item, key, list)
    f = eval(:($item -> $key)) 
    return :(lazy_map($f, $(esc(list))))
end

# TODO: this setmap_transform is restrictive since it means the user's syntax has to be invertible...
# we may be able to support a more broad class of syntaxes
function setmap_transform(item, key, list)
    f1 = eval(:($item -> $key))
    f2 = eval(:($key -> item))
    return :(lazy_bijection_set_map($f1, $f2, $(esc(list))))
end

# the first arg will be the world's name, which we don't use, since it should be in the
# already-parsed lookup_or_generate
macro _map(_, log_expr)
    _map(log_expr, :map_lookup_or_generate, map_transform)
end
# macro _setmap(_, log_expr)
#     _map(log_expr, :setmap_lookup_or_generate, setmap_transform)
# end
macro _nocollision_setmap(_, log_expr)
    _map(log_expr, :nocollision_setmap_lookup_or_generate, setmap_transform)
end

macro _dictmap(_, log_expr)
    log = GlobalRef(@__MODULE__, :lookup_or_generate)
    if (
        MacroTools.@capture(log_expr, [k_ => (addr_ ~ $log(world_[name_][key_])) for (k_, item_) in list_]) ||
        MacroTools.@capture(log_expr, (k_ => (addr_ ~ $log(world_[name_][key_])) for (k_, item_) in list_))
    )
        if name == item
            :(dictmap_lookup_or_generate($(esc(world))[$name], $(esc(list))))
        else
            f = eval(:($item -> $key))
            :(dictmap_lookup_or_generate($(esc(world))[$name], lazy_val_map($f, $(esc(list)))))
        end
    else
        error("Unexpected @dictmap expression after partial parsing: $log_expr")
    end
end

macro _origin(world, obj)
    :(lookup_or_generate($(esc(world))[:origin][$(esc(obj))]))
end

macro _index(world, obj)
    :(lookup_or_generate($(esc(world))[:index][$(esc(obj))]))
end

# TODO: should we include @concrete & @abstract?

macro _get(world, expr::Expr)
    @assert expr.head === :ref "Invalid get statement: @get $expr"
    propname = expr.args[1]
    key = Expr(:tuple, (esc(a) for a in expr.args[2:end])...)
    :(lookup_or_generate($(esc(world))[$(QuoteNode(propname))][$key]))
end

macro _arg(world, argname)
    :(lookup_or_generate($(esc(world))[:args][$(QuoteNode(argname))]))
end

const DSL_COMMANDS = Dict(
    Symbol("@$name") => Symbol("@_$name") for name in
    (:origin, :index, :arg, :get, :map, :nocollision_setmap, :dictmap, :objects)
)