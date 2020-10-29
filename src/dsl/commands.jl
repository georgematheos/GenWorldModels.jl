macro origin(world, obj)
    :(lookup_or_generate($(esc(world))[:origin][$(esc(obj))]))
end
macro origin(obj)
    :($(esc(obj)).origin) # will error if `obj` is abstract
end

macro index(world, obj)
    :(lookup_or_generate($(esc(world))[:index][$(esc(obj))]))
end
macro index(obj)
    :($(esc(obj)).idx) # will error if `obj` is abstract
end

macro concrete(world, obj)
    :(lookup_or_generate($(esc(world))[:concrete][$(esc(obj))]))
end
macro concrete(obj)
    quote
        o = $(esc(obj))
        o isa ConcreteIndexOUPMObject ? o : error("$o is not concrete; a world is needed to convert!")
    end
end

macro abstract(world, obj)
    :(lookup_or_generate($(esc(world))[:abstract][$(esc(obj))]))
end
macro abstract(obj)
    quote
        o = $(esc(obj))
        o isa AbstractOUPMObject ? o : error("$o is not abstract; a world is needed to convert!")
    end
end

macro get(world, expr::Expr)
    @assert expr.head === :ref "Invalid get statement: @get $expr"
    propname = expr.args[1]
    key = Expr(:tuple, (esc(a) for a in expr.args[2:end])...)
    :(lookup_or_generate($(esc(world))[$(QuoteNode(propname))][$key]))
end

macro arg(world, argname)
    :(lookup_or_generate($(esc(world))[:arg][$(QuoteNode(argname))]))
end

macro map_get(world, expr::Expr)
    @assert expr.head === :ref
    propname = expr.args[1]
    keys = expr.args[2]
    :(Map(lookup_or_generate)(mgfcall_map($(esc(world))[$(QuoteNode(propname))], zip($(esc(keys))))))
end

macro setmap_get(world, expr::Expr)
    @assert expr.head === :ref
    propname = expr.args[1]
    keys = expr.args[2]
    :(Map(lookup_or_generate)(mgfcall_setmap($(esc(world))[$(QuoteNode(propname))], zip($(esc(keys))))))
end

const DSL_COMMANDS = [
    Symbol("@$name") for name in
    (:origin, :index, :concrete, :abstract, :arg, :get, :map_get, :setmap_get)
]
export @origin, @index, @concrete, @abstract, @arg, @get, @map_get, @setmap_get