function WorldUpdate!(tr, objmoves::Tuple, spec::Gen.AddressTree)
    error("WorldUpdate! currently doesn't support non-dynamic constraints!  (Though this would be easy to add via a conversion to a dynamic tree.)")
end

"""
    WorldUpdate!(tr::UsingWorldTrace, objmoves::Tuple{Vararg{<:OUPMMove}}, spec::DynamicAddressTree)
    WorldUpdate!(tr::UsingWorldTrace, objmoves..., spec::DynamicAddressTree)
    WorldUpdate!(tr::UsingWorldTrace, objmoves...)

Constructs an update specification for a world model (constructed with the `@oupm` DSL)
with the given object manipulation moves and variable update specifications.
Automatically adds number statement changes to `spec`
as needed to implement the given objmoves, for every number statement for which there is not
already a constraint in `spec`.
"""
function WorldUpdate!(tr, objmoves::Tuple, spec::Gen.DynamicAddressTree)
    num_deltas = Dict()
    for objmove in objmoves
        for (sibset, delta) in num_changes_for_move(objmove)
            num_deltas[sibset] = get(num_deltas, sibset, 0) + delta
        end
    end
    for ((typename, origin), delta) in num_deltas
        num_stmt_name = num_statement_name(OriginSignature(
            typename, Tuple(typeof(obj) for obj in origin)
        ))
        # if this is not empty, ie. the user already provided a constraint, we will NOT override it
        if isempty(get_subtree(spec, :world => num_stmt_name => origin))
            current_num = tr[:world => num_stmt_name => origin => :num]
            set_subtree!(spec, :world => num_stmt_name => origin => :num, Value(current_num + delta))
        end
    end
    return WorldUpdate(objmoves, spec)
end
_provided_choicemap(args) = !isempty(args) && args[end] isa Gen.AddressTree
WorldUpdate!(tr, args...) =
    _provided_choicemap(args) ? WorldUpdate!(tr, args[1:end-1], args[end]) :
                                WorldUpdate!(tr, args, DynamicChoiceMap())

name_origin(obj::ConcreteIndexOUPMObject{T}) where {T} = (T, obj.origin)
function num_changes_for_move(move::Move)
    if move.from.origin == move.to.origin
        return ()
    else
        return (name_origin(move.from) => -1, name_origin(move.to) => +1)
    end
end
function num_changes_for_move(move::Create)
    return (name_origin(move.obj) => +1,)
end
function num_changes_for_move(move::Delete)
    return (name_origin(move.obj) => -1,)
end
function num_changes_for_move(move::Split)
    if isempty(move.moves)
        return (name_origin(move.from) => +1,)
    else
        Iterators.flatten((
            num_changes_for_sm_moves(move.moves),
            (name_origin(move.from) => +1,)
        ))
    end
end
function num_changes_for_move(move::Merge)
    if isempty(move.moves)
        return (name_origin(move.to) => -1,)
    else
        Iterators.flatten((
            num_changes_for_sm_moves(move.moves),
            (name_origin(move.to) => -1,)
        ))
    end
end 

# no constraints provided for the `from`, since the origin of these are being deleted
function num_changes_for_sm_moves(moves)
    changes = Dict()
    for (_, to) in moves
        if to !== nothing
            changes[name_origin(to)] = get(changes, name_origin(to), 0) + 1
        end
    end
    return changes
end