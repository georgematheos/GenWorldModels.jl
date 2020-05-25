mutable struct AssessWorldState <: WorldState
    constraints::ChoiceMap
    vals::Dict{Call, Any}
    weight::Float64
end
AssessWorldState(constraints) = AssessWorldState(constraints, Dict{Call, Any}(), 0.)

function begin_assess!(world::World, constraints::ChoiceMap)
    world.state = AssessWorldState(constraints)
end

function lookup_or_assess!(world::World, call::Call)
    if !haskey(world.state.vals, call)
        constraints = get_submap(world.state.constraints, addr(call) => key(call))
        gen_fn = get_gen_fn(world, call)
        weight, retval = assess(gen_fn, (world, key(call)), constraints)
        world.state.vals[call] = retval
        world.state.weight += weight
    else
        retval = world.state.vals[call]
    end
    return retval
end

function end_assess!(world::World)
    return world.state.weight
end