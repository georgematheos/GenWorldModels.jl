macro w(expr)
    memoized_fn_name = expr.args[1]
    key_expr = expr.args[2]
    return :(world[$(QuoteNode(memoized_fn_name))][key_expr])
end