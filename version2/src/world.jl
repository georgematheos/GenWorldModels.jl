#############################
# Types, constructors, gets #
#############################

"""
    Call
Represents calling the gen fn with address `addr` on argument `key`
"""
struct Call{addr}
    key
end
Call(addr, key) = Call{addr}(key)
Call(p::Pair{Symbol, <:Any}) = Call(p[1], p[2])
key(call::Call) = call.key
addr(call::Call{a}) where {a} = a

"""
    WorldState

The "state" a world is in describes how it is currently being modified,
and tracks information needed for this update.

A `NoChangeWorldState` denotes that no modifications are currently being performed.

`GenerateWorldState`, `UpdateWorldState`, and `RegenerateWorldState` all denote
that a generate, update, or regenerate is being performed on the `UsingWorld` combinator
this world is present in; these states track some metadata needed to perform
these updates on the world.
"""
abstract type WorldState end

# denotes no update or generation is currently being performed
struct NoChangeWorldState <: WorldState end

mutable struct GenerateWorldState <: WorldState
    constraints::ChoiceMap
    call_stack::Stack{Call} # stack of calls we are currently performing a lookup for
    call_sort::Vector{Call} # a topological order of the calls
    weight::Float64
end
GenerateWorldState(constraints) = GenerateWorldState(constraints, Stack{Call}(), [], 0.)

struct UpdateWorldState <: WorldState
    constraints
end
struct RegenerateWorldState <: WorldState
    constraints
end

mutable struct LookupCounts
    lookup_counts::PersistentHashMap{Call, Int} # counts[call] is the number of times this was looked up
    dependency_counts::PersistentHashMap{Call, PersistentHashMap{Call, Int}} # dependency_counts[call1][call2] = # times call2 looked up in call1
end
LookupCounts() = LookupCounts(PersistentHashMap{Call, Int}(), PersistentHashMap{Call, PersistentHashMap{Call, Int}}())

function note_new_call!(lc::LookupCounts, call::Call)
    lc.lookup_counts = assoc(lc.lookup_counts, call, 0)
    lc.dependency_counts = assoc(lc.dependency_counts, call, PersistentHashMap{Call, Int}())
end

function note_new_lookup!(lc::LookupCounts, call::Call, call_stack::Stack{Call})
    lc.lookup_counts = assoc(lc.lookup_counts, call, lc.lookup_counts[call] + 1)
    
    if !isempty(call_stack)
        looked_up_in = first(call_stack)
        new_count = get(lc.dependency_counts[call], looked_up_in, 0) + 1
        lc.dependency_counts = assoc(lc.dependency_counts, call,
            assoc(lc.dependency_counts[call], looked_up_in, new_count)
        )
    end
end

"""
    get_number_of_expected_kernel_lookups(lc::LookupCounts, call::Call)

Gives the number of traced lookups to `call` which must have occurred in the kernel
for the lookup counts object to have the number of counts it has tracked,
assuming all other calls have been properly tracked.
"""
function get_number_of_expected_kernel_lookups(lc::LookupCounts, call::Call)
    count = lc.lookup_counts[call]
    for (other_call, lookups_from_other_call) in lc.dependency_counts
        count -= get(lookups_from_other_call, call, 0)
    end
    return count
end
function get_number_of_expected_kernel_lookups(lc::LookupCounts, p::Pair)
    get_number_of_expected_kernel_lookups(lc, Call(p))
end

"""
    World
    
A `World` is a context in which generative functions can be memoized;
it stores the values which have been generated for a memoized generative functions
for each argument it has been called on.
    
Every memoized generative function for a given world has a specific address which
is used to refer to it.
"""
mutable struct World{addrs, GenFnTypes}
    gen_fns::GenFnTypes # tuple of the generative functions for this world
    state::WorldState # a state representing what operation is currently being performed on the world (eg. update, generate)
    subtraces::PersistentHashMap{Call, Trace} # subtraces[:addr => key] is the subtrace for this lookup
    lookup_counts::LookupCounts # tracks how many times each call was performed, and which calls were performed from which
    call_sort::Union{Nothing, PersistentHashMap{<:Call, Int}} # a topological sort of which calls must be finished before the next can begin
    total_score::Float64
    metadata_addr::Symbol # an address for `lookup_or_generate`s from this world in choicemaps
end

function World{addrs, GenFnTypes}(gen_fns, state, subtraces, lookup_counts, call_sort, total_score) where {addrs, GenFnTypes}
    World{addrs, GenFnTypes}(gen_fns, state, subtraces, lookup_counts, call_sort, total_score, gensym("WORLD_LOOKUP"))
end

function World{addrs, GenFnTypes}(gen_fns) where {addrs, GenFnTypes}
    World{addrs, GenFnTypes}(
        gen_fns,
        NoChangeWorldState(),
        PersistentHashMap{Call, Trace}(),
        LookupCounts(),
        nothing,
        0.
    )
end

function World(addrs::NTuple{n, Symbol}, gen_fns::NTuple{n, Gen.GenerativeFunction}) where {n}
    World{addrs, typeof(gen_fns)}(gen_fns)
end

metadata_addr(world::World) = world.metadata_addr

"""
    get_gen_fn(world::World, addr::Symbol)

Returns the generative function with address `addr` in the given world.
"""
@generated function get_gen_fn(world::World, addr_val_type::Union{Val})
    addrs, GenFnTypes = world.parameters
    addr = addr_val_type.parameters[1]
    idx = findall(addrs .== addr)[1]

    quote world.gen_fns[$idx] end
end
@inline get_gen_fn(world::World, addr::Symbol) = get_gen_fn(world, Val(addr))
@inline get_gen_fn(world::World, ::Call{addr}) where {addr} = get_gen_fn(world, addr)

has_value_for_call(world::World, call::Call) = haskey(world.subtraces, call)
total_score(world::World) = world.total_score

##############
# Generation #
##############

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

function lookup_or_generate!(world::World, call::Call)
    @assert world.state isa GenerateWorldState
    
    if !has_value_for_call(world, call)
        generate_value!(world, call)
    end

    note_new_lookup!(world.lookup_counts, call, world.state.call_stack)
    
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

#############
# ChoiceMap #
#############

struct MemoizedGenerativeFunctionChoiceMap <: Gen.ChoiceMap
    world::World
    addr::Symbol
end

Gen.has_value(c::MemoizedGenerativeFunctionChoiceMap, addr) = false
function Gen.has_value(c::MemoizedGenerativeFunctionChoiceMap, addr::Pair)
    key, rest = addr
    call = Call(c.addr, key)
    has_value(get_choices(c.world.subtraces[call]), rest)
end

function Gen.get_value(c::MemoizedGenerativeFunctionChoiceMap, addr::Pair)
    key, rest = addr
    call = Call(c.addr, key)
    get_value(get_choices(c.world.subtraces[call]), rest)
end

function Gen.get_submap(c::MemoizedGenerativeFunctionChoiceMap, addr::Pair)
    key, rest = addr
    call = Call(c.addr, key)
    get_submap(get_choices(c.world.subtraces[call]), rest)
end

Gen.get_values_shallow(c::MemoizedGenerativeFunctionChoiceMap) = ()

# TODO: store subtraces in a way s.t. we can avoid this filter call and do a direct lookup
function Gen.get_submaps_shallow(c::MemoizedGenerativeFunctionChoiceMap)
    (key(call) => get_choices(subtrace) for (call, subtrace) in filter(((call, subtrace),) -> addr(call) == c.addr, c.world.subtraces))
end

# TODO: should I use something other than a static choicemap here
# so that we can lazily construct the MemoizedGenerativeFunctionChoiceMaps
# rather than explicitly creating a tuple of them?
function Gen.get_choices(world::World{addrs, <:Any}) where {addrs}
    leaf_nodes = NamedTuple()
    internal_nodes = NamedTuple{addrs}(
        Tuple(MemoizedGenerativeFunctionChoiceMap(world, addr) for addr in addrs)
    )
    StaticChoiceMap(leaf_nodes, internal_nodes)
end