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
key(call::Call) = key(call)
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
GenerateWorldState(constraints) = GenerateWorldState(constraints, Stack(), [], 0.)

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
LookupCounts() = LookupCounts(PersistentHashMap(), PersistentHashMap())

function note_new_call!(lc::LookupCounts, call::Call)
    lc.lookup_counts = assoc(lc.lookup_counts, call, 0)
    lc.dependency_counts = assoc(lc.dependency_counts, call, PersistentHashMap())
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
    subtraces::PersistentHashMap{Call, Tr} # subtraces[:addr => key] is the subtrace for this lookup
    lookup_counts::LookupCounts # tracks how many times each call was performed, and which calls were performed from which
    call_sort::Union{Nothing, PersistentHashMap{Call, St}} # a topological sort of which calls must be finished before the next can begin
    total_score::Float64
    metadata_addr::Symbol # an address for `lookup_or_generate`s from this world in choicemaps
end

function World{addrs, GenFnTypes}(gen_fns, state, subtraces, lookup_counts, total_score)
    World{addrs, GenFnTypes}(gen_fns, state, subtraces, lookup_counts, total_score, gensym("WORLD_LOOKUP"))
end

function World{addrs, GenFnTypes}(gen_fns)
    World{addrs, GenFnTypes}
        gen_fns,
        NoChangeWorldState(),
        PersistentHashMap{Pair, Tr}(),
        LookupCounts(),
        nothing,
        0.
    )
end

function World(addrs::Tuple{Symbol}, gen_fns::NTuple{n, Gen.GenerativeFunction}) where {n}
    World{addrs, typeof(gen_fns)}(gen_fns)
end

metadata_addr(world::World) = world.metadata_addr

"""
    get_gen_fn(world::World, addr::Symbol)

Returns the generative function with address `addr` in the given world.
"""
@generated function get_gen_fn(world::World, addr_val_type::Val)
    addrs, GenFnTypes = world.parameters
    addr = addr_val_type.parameters[1]
    idx = findall(addrs .== addr)[1]

    quote world.gen_fns[$idx] end
end
@inline get_gen_fn(world::World, addr::Symbol) = get_gen_fn(world, Val(addr))
@inline get_gen_fn(world::World, ::CallTo{addr}) = get_gen_fn(world, Val(addr))

has_value(world::World, call::Call) = haskey(world.subtraces, call)

##############
# Generation #
##############

function begin_generate!(world::World, constraints::ChoiceMap)
    @assert world.state isa NoChangeWorldState "World is currently in a $(typeof(world.state)); cannot begin a generate."
    world.state = GenerateWorldState(constraints)
end

function end_generate!(world::World)
    @assert world.state isa GenerateState
    
    # TODO: check that all constraints were used
    
    # save the topological sort for the calls in the world
    world.call_sort = PersistentHashMap(call => i for (i, call) in enumerate(world.state.call_sort))
    weight = world.state.weight
    
    world.state = NoChangeWorldState()
    
    return weight
end

function lookup_or_generate!(world::World, call::Call)
    @assert world.state isa GenerateWorldState
    
    if !has_value(world, call)
        generate_value!(world, call)
    end

    note_new_lookup!(world.lookup_counts, call, world.state.call_stack)
end

function generate_value!(world, call)
    @assert world.state isa GenerateWorldState
    @assert !has_value(world, call)
    
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

Gen.get_values_shallow(c::MemoizedGenerativeFunctionChoiceMap = ()

# TODO: store subtraces in a way s.t. we can avoid this filter call and do a direct lookup
function Gen.get_submaps_shallow(c::MemoizedGenerativeFunctionChoiceMap)
    key(call) => get_choices(subtrace) for (call, subtrace) in filter(call -> addr(call) == c.addr, c.world.subtraces)
end

# TODO: should I use something other than a static choicemap here
# so that we can lazily construct the MemoizedGenerativeFunctionChoiceMaps
# rather than explicitly creating a tuple of them?
function get_choices(world::World{addrs, <:Any}) where {addrs}
    leaf_nodes = NamedTuple()
    internal_nodes = NamedTuple{addrs}(
        Tuple(MemoizedGenerativeFunctionChoiceMap(world, addr) for addr in addrs)
    )
    StaticChoiceMap(leaf_nodes, internal_nodes)
end