macro oupm(signature, body)
    sig_valid = signature isa Expr && signature.head == :call
    body_valid = body isa Expr && body.head == :block
    @assert (sig_valid && body_valid) "Invalid usage! Proper usage: @oupm model_name(args...) begin ... end"

    name = signature.args[1]
    args = signautre.args[2:end]

    expand_oupm(body, name, args)
end

struct OriginSignature
    typename::Symbol
    origin_typenames::Tuple{Vararg{Symbol}}
end

struct OUPMDSLMetaData
    type_names::Set{Symbol}
    property_names::Set{Symbol}
    number_stmts::Dict{OriginSignature, Symbol}
end
OUPMDSLMetaData() = OUPMDSLMetaData(Set(), Set(), Dict())

function expand_oupm(body, name, args)
    parse_oupm_dsl_body!(Expr[], OUPMDSLMetaData(), body)
end

function parse_oupm_dsl_body!(stmts, meta, body::Expr)
    for line in expr.args
        parse_oupm_dsl_line!(stmts, meta, line)
    end
end

# the valid body lines are:
# 1. @type ...
# 2. @property ...
# 3. @number ...
# 4. @observation_model ...
# 5. function fn_name(...) ... end / fn_name(...) = ...
function parse_oupm_dsl_line!(stmts, meta, line)
    @assert (line isa Expr) "Invalid line: $line"

    if line.head === :macrocall
        if line.args[1] == Symbol(@type)
            parse_type_line!(stmts, meta, line)
        elseif line.args[1] == Symbol(@property)
            parse_property_line!(stmts, meta, line)
        elseif line.args[1] == Symbol(@number)
            parse_number_line!(stmts, meta, line)
        elseif line.args[1] == Symbol(@observation_model)
            parse_observation_model_line!(stmts, meta, line)
        end
    else
        line = MacroTools.longdef(line)
        @assert (line.head == :function) "Invalid line: $line"
        parse_function_line!(stmst, meta, line)
    end
end

function parse_type_line!(stmts, meta, line)
    @assert MacroTools.@capture(line, @type typenames__) "Invalid type line: $line"
    for typename in typenames
        push!(stmts, :(@type typename))
        push!(meta.type_names, typename)
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
    # TODO: maybe I should actually parse the properties at a different
    # name than the one the user will use, to avoid poluting a global namespace?
    if MacroTools.@capture(line, @property name_(sig__) ~ dist_(args__))
        push!(stmts, quote
            @dist $name($sig..., ::World) = $dist($args...)
        end)
        push!(meta.property_names, name)
    elseif MacroTools.@capture(
        line,
        @property modifiers_ function name_(args__) body_ end |
        @property function name_(args__) body_ end |
        @property modifiers_ name_(args__) = body_ |
        @property name_(args__) = body_
    )
        world = gensym("world")
        # ensure calls to @origin, @get, etc., are parsed properly
        body = transform_body(body, world)
        push!(stmts,
            if modifiers !== nothing
                quote
                    @gen ($modifiers...) function $name($args...)
                        $body
                    end
                end
            else
                quote
                    @gen function $name($args...)
                        $body
                    end
                end
            end
        )
        push!(meta.property_names, name)
    else
        error("Error parsing property $line")
    end
end