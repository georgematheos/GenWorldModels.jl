##############
# Generation #
##############

mutable struct GenerateWorldState <: WorldState
    constraints::ChoiceMap
    call_stack::Stack{Call} # stack of calls we are currently performing a lookup for
    call_sort::Vector{Call} # a topological order of the calls
    weight::Float64
end
GenerateWorldState(constraints) = GenerateWorldState(constraints, Stack{Call}(), [], 0.)

# TODO: simulate

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
    world.state = GenerateWorldState(constraints)
end

"""
    end_generate!
Tell the world that we are done running generate. If `check_all_constraints_used`
is true (as it is by default), but some of the constraints provided in `begin_generate!`
were for a call which never had a value generated for it, an error will be thrown.
"""
function end_generate!(world::World; check_all_constraints_used=true)
    @assert world.state isa GenerateWorldState
    
    if check_all_constraints_used
        unused_constraints = get_ungenerated_constraints(world, world.state.constraints)
        if length(unused_constraints) != 0
            throw(error("The following constraints were not used while generating: $unused_constraints !"))
        end
    end
        
    # save the topological sort for the calls in the world
    world.call_sort = PersistentHashMap(([call => i for (i, call) in enumerate(world.state.call_sort)]::Vector{<:Pair{<:Call, Int}})...)
    weight = world.state.weight
    
    world.state = NoChangeWorldState()
    
    return weight
end

function lookup_or_generate_during_generation!(world, call)
    if !has_value_for_call(world, call)
        generate_value!(world, call)
    end

    note_new_lookup!(world.lookup_counts, call, world.stat.call_stack)

    return get_retval(world.subtraces[call])
end

function generate_value!(world, call)
    @assert world.state isa GenerateWorldState
    @assert !has_value_for_call(world, call)
    
    gen_fn = get_gen_fn(world, call)
    constraints = get_submap(world.state.constraints, addr(call) => key(call))
    
    push!(world.state.call_stack, call)
    tr, weight = generate(gen_fn, (world, key(call)), constraints)
    pop!(world.state.call_stack)
    
    push!(world.state.call_sort, call)
    
    world.subtraces = assoc(world.subtraces, call, tr)
    note_new_call!(world.lookup_counts, call)
    world.total_score += get_score(tr)
    world.state.weight += weight
end
