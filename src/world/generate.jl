#######################
# Generate / Simulate #
#######################

"""
    GenerateWorldState

World state for a call to `generate` or `simulate`.
"""
mutable struct GenerateWorldState <: WorldState
    constraints::ChoiceMap
    call_stack::Stack{Call} # stack of calls we are currently performing a lookup for
    call_sort::Vector{Call} # a topological order of the calls
    weight::Float64
    generate_fn::Function # will determine whether we use `generate` or `simulate`
end
GenerateWorldState(constraints, generate_fn) = GenerateWorldState(constraints, Stack{Call}(), [], 0., generate_fn)

generate_fn(s::GenerateWorldState) = s.generate_fn

"""
    get_ungenerated_constraints(world, constraints)
    
Returns a list of all the Calls which constraints were provided for, but which
do not have a value generated for them in the world.
"""
function get_ungenerated_constraints(world::World, constraints::ChoiceMap)
    unused_constraints = []
    for (address, key_to_submap) in get_submaps_shallow(constraints)
        for (key, submap) in get_submaps_shallow(key_to_submap)
            c = Call(address, key)
            if !has_value_for_call(world, c)
                push!(unused_constraints, c)
            end
        end
    end
    return unused_constraints
end

"""
    begin_generate!(world, constraints)

Tell the world we are about to run generate on the kernel.
`get_submap(constraints, addr => key)` contains the constraints
which should be used to generate the value for call `(addr, key)`.
All calls provided in the call must have a value generated for them
before generation finishes (as denoted by calling `end_generate!`).
"""
function begin_generate!(world::World, constraints::ChoiceMap)
    @assert (world.state isa NoChangeWorldState) "World is currently in a $(typeof(world.state)); cannot begin a generate."
    world.state = GenerateWorldState(constraints, Gen.generate)
end

function begin_simulate!(world::World)
    @assert (world.state isa NoChangeWorldState) "World is currently in a $(typeof(world.state)); cannot begin a simulate."
    world.state = GenerateWorldState(EmptyChoiceMap(), (gen_fn, args, constraints) -> (Gen.simulate(gen_fn, args), 0.))
end

"""
    end_generate!
Tell the world that we are done running generate. If `check_all_constraints_used`
is true (as it is by default), but some of the constraints provided in `begin_generate!`
were for a call which never had a value generated for it, an error will be thrown.
"""
function end_generate!(world::World, check_all_constraints_used)
    @assert world.state isa GenerateWorldState
    
    if check_all_constraints_used
        unused_constraints = get_ungenerated_constraints(world, world.state.constraints)
        if length(unused_constraints) != 0
            throw(error("The following constraints were not used while generating: $unused_constraints !"))
        end
    end
        
    # save the topological sort for the calls in the world
    world.call_sort = CallSort(world.state.call_sort)
    weight = world.state.weight
    
    world.state = NoChangeWorldState()
    
    return weight
end
end_simulate!(args...) = end_generate!(args...)

function lookup_or_generate_during_generate!(world, call)
    if !has_value_for_call(world, call)
        generate_value_during_generate!(world, call)
    end

    note_new_lookup!(world, call, world.state.call_stack)

    return get_val(world.calls, call)
end

function generate_value_during_generate!(world, call)
    constraints = get_submap(world.state.constraints, addr(call) => key(call))
    
    push!(world.state.call_stack, call)
    weight = generate_value!(world, call, constraints)
    pop!(world.state.call_stack)
    
    push!(world.state.call_sort, call)

    world.state.weight += weight
end