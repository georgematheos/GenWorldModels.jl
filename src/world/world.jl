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

"""
    LookupCounts

An immutable datatype to track the dependencies in the world and the number of times each lookup has occurred.
"""
struct LookupCounts
    lookup_counts::PersistentHashMap{Call, Int} # counts[call] is the number of times this was looked up
    dependency_counts::PersistentHashMap{Call, PersistentHashMap{Call, Int}} # dependency_counts[call1][call2] = # times call2 looked up in call1
end
LookupCounts() = LookupCounts(PersistentHashMap{Call, Int}(), PersistentHashMap{Call, PersistentHashMap{Call, Int}}())

function note_new_call(lc::LookupCounts, call::Call)
    new_lookup_counts = assoc(lc.lookup_counts, call, 0)
    new_dependency_counts = assoc(lc.dependency_counts, call, PersistentHashMap{Call, Int}())
    LookupCounts(new_lookup_counts, new_dependency_counts)
end

function note_new_lookup(lc::LookupCounts, call::Call, call_stack::Stack{Call})
    new_lookup_counts = assoc(lc.lookup_counts, call, lc.lookup_counts[call] + 1)    
    if !isempty(call_stack)
        looked_up_in = first(call_stack)
        new_count = get(lc.dependency_counts[looked_up_in], call, 0) + 1
        new_dependency_counts = assoc(lc.dependency_counts, looked_up_in,
            assoc(lc.dependency_counts[looked_up_in], call, new_count)
        )
    else
        new_dependency_counts = lc.dependency_counts
    end
    LookupCounts(new_lookup_counts, new_dependency_counts)
end
# note lookup removal from another call (ie. not from kernel)
function note_lookup_removed(lc::LookupCounts, call::Call, called_from::Call)
    new_count = lc.dependency_counts[called_from][call] - 1
    if new_count == 0
        new_dependency_counts = assoc(lc.dependency_counts, called_from,
            dissoc(lc.dependency_counts[called_from], call)
        )
    else
        new_dependency_counts = assoc(lc.dependency_counts, called_from,
        assoc(lc.dependency_counts[called_from], call, new_count)
        )
    end

    new_count = lc.lookup_counts[call] - 1
    if new_count == 0
        new_lookup_counts = dissoc(lc.lookup_counts, call)
    else
        new_lookup_counts = assoc(lc.lookup_counts, call, new_count)
    end

    (LookupCounts(new_lookup_counts, new_dependency_counts), new_count)
end
# note lookup removed from kernel (ie. not removed from another call)
function note_lookup_removed(lc::LookupCounts, call::Call)
    new_count = lc.lookup_counts[call] - 1
    if new_count == 0
        new_lookup_counts = dissoc(lc.lookup_counts, call)
    else
        new_lookup_counts = assoc(lc.lookup_counts, call, new_count)
    end

    (LookupCounts(new_lookup_counts, lc.dependency_counts), new_count)
end

function get_calls_looked_up_in(lc::LookupCounts, call)
    (looked_up_in for (looked_up_in, count) in lc.dependency_counts[call] if count > 0)
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
    CallSort

A data structure to store a topological sort for all the calls in the world.
If the algorithms work properly, at the end of every update, the call sort
and the lookup counts should be consistent in that the call sort should
have a valid topological ordering for the dependencies between the calls tracked
in lookup counts.
"""
struct CallSort
    call_to_idx::PersistentHashMap
    max_index::Int
end
function CallSort(idx_to_call::Vector{Call})
    call_to_idx = PersistentHashMap(([call => i for (i, call) in enumerate(idx_to_call)]))
    CallSort(call_to_idx, length(idx_to_call))
end
Base.getindex(srt::CallSort, call::Call) = srt.call_to_idx[call]
function change_index_to(srt::CallSort, call::Call, idx::Int)
    new_map = assoc(srt.call_to_idx, call, idx)
    new_max_idx = max(srt.max_index, idx)
    CallSort(new_map, new_max_idx)
end
add_call_to_end(srt::CallSort, call::Call) = change_index_to(srt, call, srt.max_index + 1)

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
    call_sort::CallSort # a topological sort of which calls must be finished before the next can begin
    total_score::Float64
    metadata_addr::Symbol # an address for `lookup_or_generate`s from this world in choicemaps
end

function World{addrs, GenFnTypes}(gen_fns, state, subtraces, lookup_counts, call_sort, total_score) where {addrs, GenFnTypes}
    World{addrs, GenFnTypes}(gen_fns, state, subtraces, lookup_counts, call_sort, total_score, gensym("WORLD_LOOKUP"))
end

World(w::World{A,G}) where {A,G} = World{A,G}(w.gen_fns, w.state, w.subtraces, w.lookup_counts, w.call_sort, w.total_score, w.metadata_addr)

function World{addrs, GenFnTypes}(gen_fns) where {addrs, GenFnTypes}
    World{addrs, GenFnTypes}(
        gen_fns,
        NoChangeWorldState(),
        PersistentHashMap{Call, Trace}(),
        LookupCounts(),
        CallSort(Call[]), # will be overwritten after `generate`
        0.
    )
end

function World(addrs::NTuple{n, Symbol}, gen_fns::NTuple{n, Gen.GenerativeFunction}) where {n}
    World{addrs, typeof(gen_fns)}(gen_fns)
end

metadata_addr(world::World) = world.metadata_addr

# functions for tracking counts and dependency structure
function note_new_call!(world, args...)
    world.lookup_counts = note_new_call(world.lookup_counts, args...)
end
function note_new_lookup!(world, args...)
    world.lookup_counts = note_new_lookup(world.lookup_counts, args...)
end
function note_lookup_removed!(world, args...)
    (world.lookup_counts, new_num) = note_lookup_removed(world.lookup_counts, args...)
    new_num
end

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

get_calls_looked_up_in(world::World, call) = get_calls_looked_up_in(world.lookup_counts, call)
has_value_for_call(world::World, call::Call) = haskey(world.subtraces, call)
get_value_for_call(world::World, call::Call) = get_retval(world.subtraces[call])
get_trace(world::World, call::Call) = world.subtraces[call]
total_score(world::World) = world.total_score

"""
    generate_value!(world, call, constraints)

This will call `Gen.generate` for the given call with the given constraints,
and add the resulting trace to the world.
This function handles tracking all world-level tracking, but does not
handle tracking relating to the specific `world.state` (eg. it does not
increase the world update state weight.)
Returns the `weight` returned by the call to `Gen.generate`
"""
function generate_value!(world, call, constraints)
    @assert !(world.state isa NoChangeWorldState)
    @assert !has_value_for_call(world, call)
    
    gen_fn = get_gen_fn(world, call)
    tr, weight = generate(gen_fn, (world, key(call)), constraints)
    world.subtraces = assoc(world.subtraces, call, tr)
    note_new_call!(world, call)
    world.total_score += get_score(tr)

    return weight
end

"""
    lookup_or_generate!

This is the "interface" with the `lookup_or_generate` generative function;
whenever a new call to `generate` is made for `lookup_or_generate`,
or whenever `update` is called on `lookup_or_generate` in such a way that
it could alter the dependency structure/lookup counts of the world, `lookup_or_generate!`
should be called.
Will return the value that the given `call` returns after the generation/execution.
"""
function lookup_or_generate!(world::World, call::Call)    
    # if somehow we called a `generate` while running updates,
    # we need to handle it differently
    if world.state isa UpdateWorldState
        lookup_or_generate_during_update!(world, call)
    elseif world.state isa GenerateWorldState
        lookup_or_generate_during_generate!(world, call)
    else
        error("World must be in a Generate or Update state if `lookup_or_generate!` is called.")
    end
end

include("generate.jl")
include("choicemap.jl")
include("update.jl")