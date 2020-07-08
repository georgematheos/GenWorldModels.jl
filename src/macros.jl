"""
    @w memoized_gen_fn[lookup_key]

Syntactic sugar for `lookup_or_generate(world[:memoized_gen_fn][lookup_key])`.

Assumes a variable `world` exists in the current scope and `lookup_or_generate`
is defined at the global scope.
"""
macro w(expr)
    memoized_fn_name = expr.args[1]
    key_expr = expr.args[2]
    return :($(esc(:world))[$(QuoteNode(memoized_fn_name))][$key_expr])
end

"""
    @WorldMap(memoized_gen_fn_addr, lookup_key_iterable)

Looks up each key in `lookup_key_iterable` in `world[memoized_gen_fn_addr]`.
This macroexpands into a call to `Map(lookup_or_generate)`
called on `[world[memoized_gen_fn_addr][key] for key in lookup_key_iterable]`.

Assumes `world` exists in the current scope and `Map` is defined at the global scope.
"""
macro WorldMap(mgf_addr, args)
    lookup_or_generate = esc(:($(@__MODULE__).lookup_or_generate))
    quote
        Map($lookup_or_generate)([
            $(esc(:world))[$mgf_addr][arg] for arg in $args
        ])
    end
end

"""
Parses the expressions for memoized generative functions.
Converts `gen_fn` to `:gen_fn => gen_fn`;
does not modify `:address => gen_fn`.
"""
function parse_mgf_exprs(mgf_raw_exprs)
    mgf_parsed_exprs = [
        if expr isa Expr
            @assert expr.head == :call && expr.args[1] == :(=>)
            expr
        else
            @assert expr isa Symbol
            :($(QuoteNode(expr))=>$expr)
        end
        for expr in mgf_raw_exprs
    ]
end

"""
    @UsingWorld(kernel, memoized_gen_fns...[; world_args=[]])

# Arguments
- `kernel::GenerativeFunction`: The kernel generative function for the `UsingWorld` model
- `memoized_gen_fns::Vector...`: A list of generative functions to be memoized in the `UsingWorld`
model.  Each is specified either as `:address => gen_fn`, in which case `:address` is used
as the address for this memoized generative function, or is specified as `gen_fn`, in which case
`:gen_fn` is used as the address.

# Optional Keyword Arguments
- `world_args::Vector{Symbol}`: A list of symbols for the names assigned to
the first `length(world_args)` arguments to the resulting `UsingWorld` generative function.
These arguments are not passed into the kernel, and are instead stored in the world,
to be accessed via `getarg(world, :arg_name)`.

Assumes that the kernel and memoized generative functions are defined at the global scope.
"""
macro UsingWorld(exprs...)
    @assert length(exprs) > 0 "Invalid usage!"

    has_kwargs = exprs[1] isa Expr && exprs[1].head == :parameters
    using_world_args = Vector{Union{Expr, Symbol}}()

    if has_kwargs
        push!(using_world_args, exprs[1])
        exprs = exprs[2:end]
    end

    kernel_expr = exprs[1]
    mgf_exprs = parse_mgf_exprs(exprs[2:end])

    push!(using_world_args, kernel_expr)
    push!(using_world_args, mgf_exprs...)

    using_world = esc(:($(@__MODULE__).UsingWorld))

    quote $using_world($(using_world_args...)) end
end