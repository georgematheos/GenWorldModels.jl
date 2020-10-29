import MacroTools

include("commands.jl")

macro oupm(signature, body)
    sig_valid = signature isa Expr && signature.head == :call
    body_valid = body isa Expr && body.head == :block
    @assert (sig_valid && body_valid) "Invalid usage! Proper usage: @oupm model_name(args...) begin ... end"

    name = signature.args[1]
    args = signature.args[2:end]

    expand_oupm(body, name, args, __module__)
end

struct OriginSignature
    typename::Symbol
    origin_typenames::Tuple{Vararg{Symbol}}
end

mutable struct OUPMDSLMetaData
    model_name::Symbol
    model_args::Tuple{Vararg{Symbol}}
    type_names::Set{Symbol}
    property_names::Set{Symbol}
    number_stmts::Dict{OriginSignature, Symbol}
    observation_model_name::Union{Nothing, Symbol}
end

OUPMDSLMetaData(model_name, model_args) = OUPMDSLMetaData(model_name, model_args, Set(), Set(), Dict(), nothing)

function expand_oupm(body, name, args, __module__)
    meta = OUPMDSLMetaData(name, Tuple(args))
    stmts = Union{LineNumberNode, Expr}[] 
    parse_oupm_dsl_body!(stmts, meta, body)

    # due to https://github.com/JuliaLang/julia/issues/37691, we cannot escape names within the
    # shallow macroexpansion provided by this DSL, and then return that expression and trust
    # the normal macroexpansion system to work.  The issue is that the `@gen` macro, etc.,
    # do not expect escaping to occur in the expressions they parse.
    # Instead, we do not add escaping within our macros, and expand here, then
    # add escaping after the fact.
    stmts = [esc(macroexpand(__module__, stmt)) for stmt in stmts]
    
    return quote
        $(stmts...)
        $(esc(meta.model_name)) = UsingWorld(
            $(esc(meta.observation_model_name)),
            $((:($(QuoteNode(name)) => $(esc(name))) for name in meta.property_names)...),
            $((:($(QuoteNode(name)) => $(esc(name))) for (_, name) in meta.number_stmts)...);
            world_args=$(meta.model_args)
        )
    end
end

function parse_oupm_dsl_body!(stmts, meta, body::Expr)
    for line in body.args
        parse_oupm_dsl_line!(stmts, meta, line)
    end
end

# the valid body lines are:
# 1. @property ...
# 2. @number ...
# 3. @observation_model ...
parse_oupm_dsl_line!(stmts, meta, ln::LineNumberNode) = push!(stmts, ln)
function parse_oupm_dsl_line!(stmts, meta, line)
    @assert (line isa Expr && line.head === :macrocall) "Invalid line: $line"

    if line.args[1] == Symbol("@property")
        parse_property_line!(stmts, meta, line)
    elseif line.args[1] == Symbol("@number")
        parse_number_line!(stmts, meta, line)
    elseif line.args[1] == Symbol("@observation_model")
        parse_observation_model!(stmts, meta, line)
    else
        error("Invalid line: $line")
    end
end

#=
Possibilities:
1. @property name(::..., ::..., ...) ~ dist(...)
2. @property (modifiers...) function name(::..., ::...)
    ...
end
=#
function parse_property_line!(stmts, meta, line)
    if MacroTools.@capture(line, @property name_(sig__) ~ dist_(args__))
        (names, types) = parse_sig(sig)
        push!(stmts,
            :( @dist $name(::World, $(Expr(:tuple, names...))::Tuple{$(types...)}) = $dist($(args...)) )
        )

        push!(meta.property_names, name)
    elseif MacroTools.@capture(
        line,
        (@property modifiers_ function name_(args__) body_ end) |
        (@property function name_(args__) body_ end) |
        (@property modifiers_ name_(args__) = body_) |
        (@property name_(args__) = body_)
    )
        world = gensym("world")
        # ensure calls to @origin, @get, etc., are parsed properly
        body = parse_world_into_and_trace_commands(body, world)
        (names, types) = parse_sig(args)

        fndef = :(
            function $name($world::World, $(Expr(:tuple, names...))::Tuple{$(types...)})
                $(body.args...)
            end
        )
        linenum = get_macrocall_line_num(line)
        lastargs = modifiers === nothing ? (fndef,) : (modifiers, fndef)

        # @gen (modifiers...) function $body end
        push!(stmts, Expr(:macrocall, Symbol("@gen"), linenum, lastargs...))
        push!(meta.property_names, name)
    else
        error("Error parsing property $line")
    end
end

function get_macrocall_line_num(line)
    if !(line isa Expr)
        return nothing
    end
    if line.head === :block
        if length(line.args) > 1
            return nothing
        end
        return get_macrocall_line_num(line.args[1])
    end
    if line.head === :macrocall
        return line.args[2]
    end
end

#=
Supported constructs:
1. @number Type(sig...) ~ dist()
2. @number (annotations...) function Type(sig...)
        ...
    end
=#
function parse_number_line!(stmts, meta, line)
    if MacroTools.@capture(line, @number name_(sig__) ~ dist_(args__))
        (names, types) = parse_sig(sig)
        linenum = get_macrocall_line_num(line)
        fn_expr = fn_name -> Expr(:macrocall, Symbol("@gen"), linenum, :((static, diffs)), :(
            function $fn_name(::World, $(Expr(:tuple, names...))::Tuple{$(types...)})
                num ~ $dist($(args...))
                return num
            end
        ))
    elseif MacroTools.@capture(
        line,
        (@number modifiers_ function name_(sig__) body_ end) |
        (@number function name_(sig__) body_ end) |
        (@number modifiers_ name_(sig__) = body_) |
        (@number name_(sig__) = body_)
    )
        world = gensym("world")
        linenum = get_macrocall_line_num(line)
        body = parse_world_into_and_trace_commands(body, world)
        (names, types) = parse_sig(sig)
        fn_expr = let w = world, tags = (modifiers !== nothing) ? (linenum, modifiers) : (linenum,)
            fn_name -> Expr(:macrocall, Symbol("@gen"), tags..., :(
                function $fn_name($w::World, $(Expr(:tuple, names...))::Tuple{$(types...)}) $(body.args...) end
            ))
        end
    else
        error("Unrecognized @number construct: $line")
    end
    origin_sig = parse_origin_sig(name, sig)
    fn_name = get_num_statement_name(origin_sig.typename, origin_sig.origin_typenames)
    
    push!(stmts, fn_expr(fn_name))
    meta.number_stmts[origin_sig] = fn_name
end

function get_num_statement_name(typename, origin_typenames)
    root = "num_" * String(typename) * "__" *((String(n)*"_" for n in origin_typenames)...)
    gensym(root)
end

function parse_origin_sig(name, sig)
    @assert all(expr.head === :(::) for expr in sig) "Invalid origin signature: $name($(sig...))"
    types = Tuple(last(expr.args) for expr in sig)
    OriginSignature(name, types)
end

function parse_sig(sig)
    function name_type(expr)
        if expr isa Expr && expr.head === :(::)
            if length(expr.args) == 2
                Tuple(expr.args)
            else
                (:_, expr.args[1])
            end
        else
            @assert expr isa Symbol
            (expr, Any)
        end
    end
    name_types = (name_type(expr) for expr in sig)
    names = Tuple(name for (name, _) in name_types)
    types = Tuple(type for (_, type) in name_types)
    return (names, types)
end

function parse_observation_model!(stmts, meta, line)
    @assert meta.observation_model_name === nothing "There should only be one observatin model!"
    if MacroTools.@capture(
        line,
        (@observation_model modifiers_ function name_(sig__) body_ end) |
        (@observation_model function name_(sig__) body_ end) |
        (@observation_model modifiers_ name_(sig__) = body_) |
        (@observation_model name_(sig__) = body_)
    )
        meta.observation_model_name = name
        world = gensym("world")
        body = parse_world_into_and_trace_commands(body, world)
        linenum = get_macrocall_line_num(line)
        tags = modifiers === nothing ? (linenum,) : (linenum, modifiers)

        push!(stmts, Expr(:macrocall, Symbol("@gen"), tags..., :(
                function $name($world::World, $(sig...))
                    $(body.args...)
                end
            )
        ))
    else
        error("Failed to parse observation model: $line")
    end
end

"""
    parse_world_into_and_trace_commands(body, worldname)

Transform the expression `body` so that every special command which is called
gets access to the world, and is called in a traced expression.
Roughly, we transform `...@command(args...)...` to `...({gensym()} ~ @command(world, args...)))...`.
"""
function parse_world_into_and_trace_commands(body::Expr, worldname::Symbol)
    MacroTools.postwalk(body) do e
        if e isa Expr && e.head === :macrocall && e.args[1] in DSL_COMMANDS
            name = gensym(String(e.args[1]) * "_call_result")
            if length(e.args) > 1 && e.args[2] isa LineNumberNode
                new_expr = Expr(:macrocall, e.args[1:2]..., worldname, e.args[3:end]...)
            else
                new_expr = Expr(:macrocall, e.args[1], worldname, e.args[2:end]...)
            end
            :(({$(QuoteNode(name))} ~ $new_expr))
        else
            e
        end
    end
end