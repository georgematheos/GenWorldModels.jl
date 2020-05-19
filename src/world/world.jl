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

function lookup_or_generate!(world::World, call::Call)    
    # if somehow we called a `generate` while running updates,
    # we need to handle it differently
    if world.state isa Union{UpdateWorldState, RegenerateWorldState}
        lookup_or_generate_during_update!(world, call)
    elseif world.state isa GenerateState
        lookup_or_generate_during_generation!(world, call)
    else
        error("World must be in a Generate, Update, or Regenerate state if `lookup_or_generate!` is called.")
    end
end

include("generate.jl")
include("choicemap.jl")
include("update.jl")