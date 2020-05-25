mutable struct ProposeWorldState <: WorldState
    choices::DynamicChoiceMap
    vals::Dict{Call, Any}
    weight::Float64
end
ProposeWorldState() = ProposeWorldState(choicemap(), Dict{Call, Any}(), 0.)

function begin_propose!(world::World)
    world.state = ProposeWorldState()
end

function lookup_or_propose!(world::World, call::Call)
    if !haskey(world.state.vals, call)
        gen_fn = get_gen_fn(world, call)
        choices, weight, retval = propose(gen_fn, (world, key(call)))
        world.state.vals[call] = retval
        set_submap!(world.state.choices, addr(call) => key(call), choices)
        world.state.weight += weight
    else
        retval = world.state.vals[call]
    end
    return retval
end

function end_propose!(world::World)
    return (world.state.choices, world.state.weight)
end