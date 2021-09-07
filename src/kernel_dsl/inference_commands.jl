"""
    parse_siblingset_sig(:(Typename(origin_1, ..., origin_n)))

Returns `(:Typename, :(origin_1, ..., origin_n), :(type_1, ..., type_n))`
where `type_i` evaluates to the type of `origin_i`.
"""
parse_siblingset_sig(sig) =
    if MacroTools.@capture(sig, Typename_(origins__))
        (
            Typename,
            Expr(:tuple, origins...),
            Expr(:tuple, 
                (:($(GlobalRef(@__MODULE__, :oupm_type_name))($obj))
                for obj in origins)...)
        )
    else
        @error("Invalid `@get_number(sig) signature: $sig")
    end

"""
    @get_number(tr, Typename(origin_1, ..., origin_n))

Get the number of objects under the world trace of type `Typename` with
the given origin.
"""
macro get_number(tr, sig)
    (typename, origins, types) = parse_siblingset_sig(sig)
    num_stmt_name = :(
        num_statement_name(OriginSignature($(QuoteNode(typename)), $(esc(types))))
    )
    :($(esc(tr))[:world => $num_stmt_name => $(esc(origins)) => :num])
end

"""
    @set_number(Typename(origin_1, ..., origin_n), new_num)

Expands to a tuple `(addr, new_num)` where `addr` is the address of
the number of objects matching `Typename(origin_1, ..., origin_n)` in a world model.
Can be used within choicemaps to constrain this address.

### Example
```julia
generate(world_model, args, choicemap(
    @set_number(Suits(), 4),
    @set_number(Cards(Suit(1)), 13)
))
```
"""
macro set_number(sig, newnum)
    typename, origins, types = parse_siblingset_sig(sig)
    n_stmt = :(num_statement_name(OriginSignature($(QuoteNode(typename)), $(esc(types)))))
    return :((:world => $n_stmt => $(esc(origins)) => :num, $(esc(newnum))))
end

key(objs) = Expr(:tuple, [esc(obj) for obj in objs]...)
"""
    @get(tr, prop_sigs...)

Where each element of `prop_sigs` takes the form
- prop[objs...] => addr
or
- prop[objs...]

Get the values of properties, or values at addresses within property traces, in a world trace.

### Examples
```julia
val1, val2, val3 = @get(tr, prop1[obj1], prop2[obj2a, obj2b] => :addr, prop3[])
val = @get(tr, prop[obj1, obj2] => addr)
```
"""
macro get(tr, exprs...)
    getexprs = Expr[]
    for expr in exprs
        push!(getexprs, :($(esc(tr))[$(_addr(expr, "@get"))]))
    end

    length(exprs) > 1 ? Expr(:tuple, getexprs...) : getexprs[1]
end

"""
    @set(prop[objs...] => addr, val)
    @set(props[objs...], val)

Expands into a tuple `(addr, val)` where `addr` is the address to constrain
the given property in a world model.

### Example
```julia
generate(world_model, args, choicemap(
    @set(color[Ocean(1)], :blue),
    @set(proximity[Ocean(1), Ocean(2)] => :are_bordering, true)
))
```
"""
macro set(sig, val)
    :(($(_addr(sig, "@set")), $(esc(val))))
end

"""
    @addr(prop_sig)

Where prop_sig is either
- prop[objs...] => addr
or
- prop[objs...]

Get the Gen address of the given property, or the address of the given choice within a property trace.

### Examples
```julia
value = tr[@addr(prop1[obj1] => addr)]
# equivalent to `value = @get(tr, prop1[obj1] => addr)`
```
"""
macro addr(expr)
    _addr(expr, "@addr")
end

function _addr(expr, command)
    if MacroTools.@capture(expr, propname_[objs__] => addr_)
        return :(:world => $(QuoteNode(propname)) => $(key(objs)) => $(esc(addr)))
    elseif MacroTools.@capture(expr, propname_[objs__])
        return :(:world => $(QuoteNode(propname)) => $(key(objs)))
    else
        error("Invalid property spec in $command command: $expr")
    end
end

"""
    @obsmodel()

The address of the a world model's observation model.

### Example
```julia
num_samples_drawn = tr[@obsmodel() => :num_samples]
```
"""
macro obsmodel()
    :(:kernel)
end

# TODO: docstrings
macro abstract(tr, objexpr)
    :(convert_to_abstract(get_world($(esc(tr))), $(esc(objexpr))))
end
macro concrete(tr, objexpr)
    :(convert_to_concrete(get_world($(esc(tr))), $(esc(objexpr))))
end
macro index(tr, objexpr)
    :(get_val(get_world($(esc(tr))), Call(_get_index_addr, $(esc(objexpr)))))
end
macro origin(tr, objexpr)
    let origin = :(get_val(get_world($(esc(tr))), Call(_get_origin_addr, $(esc(objexpr)))))
            :(Tuple(
                convert_to_concrete(get_world($(esc(tr))), obj) for obj in $origin
            ))
    end
end

"""
    @arg(tr, arg_name_symbol)

Get the world argument named `arg_name_symbol` in the world trace.

### Example
```julia
@arg(tr, ξ)
```
"""
macro arg(tr, sym)
    :(
        get_world_args($(esc(tr)))[$(QuoteNode(sym))]
    )
end

macro regenerate(expr)
    :(($(_addr(expr, "@regenerate")), $(GlobalRef(Gen, :AllSelection)())))
end

include("objectsets.jl")