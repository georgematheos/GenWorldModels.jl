function get_types(sig)
    if MacroTools.@capture(sig, Typename_(origins__))
        types = :(Tuple(
            oupm_type_name(objexpr)
            for objexpr in $origins
        ))
        return (Typename, types)
    else
        @error("Invalid `@get_number(sig) signature: $sig")
    end
end

# TODO: docstrings

macro get_number(tr, sig)
    (typename, types) = get_types(sig)
    num_stmt_name = :(
        num_statement_name(OriginSignature($typename, $types))
    )
    :($tr[:world => $num_stmt_name => :num])
end

key(objs) = Expr(:tuple, [esc(obj) for obj in objs]...)
macro get(tr, exprs...)
    getexprs = Expr[]
    for expr in exprs
        if MacroTools.@capture(expr, propname_[objs__] => addr_)
            getexpr = :($(esc(tr))[:world => $(QuoteNode(propname)) => $(key(objs)) => $(esc(addr))])
        elseif MacroTools.@capture(expr, propname_[objs__])
            getexpr = :($(esc(tr))[:world => $(QuoteNode(propname)) => $(key(objs))])
        else
            error("Invalid get statement: @get(tr, ..., $expr, ...)")
        end
        push!(getexprs, getexpr)
    end

    Expr(:tuple, getexprs...)
end

macro set(expr)
    if MacroTools.@capture(expr, prop_[objs__] = val_)
        :((:world => $(QuoteNode(prop)) => $(key(objs)), $(esc(val))))
    elseif MacroTools.@capture(expr, prop_[objs__] => addr_ = val_)
        :((:world => $(QuoteNode(prop)) => $(key(objs)) => $(esc(addr)), $(esc(val))))
    else
        error("Invalid set expression: @set $expr")
    end
end

macro obsmodel()
    :(:kernel)
end

macro index(tr, objexpr)
    :(get_val($(esc(tr)).world, Call(_get_index_addr, $(esc(objexpr)))))
end
macro abstract(tr, objexpr)
    :(convert_to_abstract($(esc(tr)).world, $(esc(objexpr))))
end
macro concrete(tr, objexpr)
    :(convert_to_concrete($(esc(tr)).world, $(esc(objexpr))))
end
macro origin(tr, objexpr)
    :(get_val($(esc(tr)).world, Call(_get_origin_addr, $(esc(objexpr)))))
end

include("objectsets.jl")