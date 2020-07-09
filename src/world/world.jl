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

include("call.jl") # `Call` type
include("calls.jl") # `Calls` data structure to store subtraces
include("lookup_counts.jl") # `LookupCounts` data structure to store the dependency graph & the counts
include("call_sort.jl") # `CallSort` data structure to maintain a topological numbering of the calls.

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
    calls::Calls
    lookup_counts::LookupCounts # tracks how many times each call was performed, and which calls were performed from which
    call_sort::CallSort # a topological sort of which calls must be finished before the next can begin
    total_score::Float64
    metadata_addr::Symbol # an address for `lookup_or_generate`s from this world in choicemaps
end

function World{addrs, GenFnTypes}(gen_fns, state, calls, lookup_counts, call_sort, total_score) where {addrs, GenFnTypes}
    World{addrs, GenFnTypes}(gen_fns, state, calls, lookup_counts, call_sort, total_score, gensym("WORLD_LOOKUP"))
end

World(w::World{A,G}) where {A,G} = World{A,G}(w.gen_fns, w.state, w.calls, w.lookup_counts, w.call_sort, w.total_score, w.metadata_addr)

function World{addrs, GenFnTypes}(gen_fns, world_args::NamedTuple) where {addrs, GenFnTypes}
    w = World{addrs, GenFnTypes}(
        gen_fns,
        NoChangeWorldState(),
        Calls(addrs, gen_fns, world_args),
        LookupCounts(),
        CallSort(Call[]), # will be overwritten after `generate`
        0.
    )
    for (arg_address, _) in pairs(world_args)
        note_new_call!(w, Call(_world_args_addr, arg_address))
    end
    return w
end

function World(addrs::NTuple{n, Symbol}, gen_fns::NTuple{n, Gen.GenerativeFunction}, world_args::NamedTuple) where {n}
    World{addrs, typeof(gen_fns)}(gen_fns, world_args)
end

# TODO: Could make this more informative by showing what calls it has in it
Base.show(io::IO, world::World{addrs, <:Any}) where {addrs} = print(io, "world{$addrs}")

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
function remove_call_from_sort!(world, call)
    world.call_sort = remove_call(world.call_sort, call)
end

"""
    get_gen_fn(world::World, addr::Symbol)

Returns the generative function with address `addr` in the given world.
"""
@generated function get_gen_fn(world::World, addr_val_type::Union{Val})
    addrs, GenFnTypes = world.parameters
    addr = addr_val_type.parameters[1]
    all_inds = findall(addrs .== addr)
    if length(all_inds) == 0
        throw(KeyError(addr))
    elseif length(all_inds) > 1
        error("Multiple memoized generative functions detected for address $addr")
    end
    idx = all_inds[1]

    quote world.gen_fns[$idx] end
end
@inline get_gen_fn(world::World, addr::Symbol) = get_gen_fn(world, Val(addr))
@inline get_gen_fn(world::World, ::Call{addr}) where {addr} = get_gen_fn(world, addr)

get_all_calls_which_look_up(world::World, call) = get_all_calls_which_look_up(world.lookup_counts, call)
has_value_for_call(world::World, call::Call) = has_val(world.calls, call)
get_value_for_call(world::World, call::Call) = get_val(world.calls, call)
get_trace(world::World, call::Call) = get_trace(world.calls, call)
total_score(world::World) = world.total_score

"""
    generate_value!(world, call, constraints)

This will call generate for the given call with the given constraints,
and add the resulting trace to the world.
This function handles tracking all world-level tracking, but does not
handle tracking relating to the specific `world.state` (eg. it does not
increase the world update state weight.)

Generation will occur via the function specified by `generate_fn(world.state)`;
this function should have the same signature as `Gen.generate`.
(For `simulate`, one can set `generate_fn = (gen_fn, args, constraints) -> (Gen.simulate(gen_fn, args), 0.)`)

Returns the `weight` returned by the call to the `generate` (or whatever function was called instead).
"""
function generate_value!(world, call, constraints)
    @assert !(world.state isa NoChangeWorldState)
    @assert !has_value_for_call(world, call)

    gen_fn = get_gen_fn(world, call)
    tr, weight = generate_fn(world.state)(gen_fn, (world, key(call)), constraints)
    world.calls = assoc(world.calls, call, tr)
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
function lookup_or_generate!(world::World, call::Call; reason_for_call=:generate)    
    # if somehow we called a `generate` while running updates,
    # we need to handle it differently
    if world.state isa UpdateWorldState
        lookup_or_generate_during_update!(world, call, reason_for_call)
    elseif world.state isa GenerateWorldState
        lookup_or_generate_during_generate!(world, call)
    else
        error("World must be in a Generate or Update state if `lookup_or_generate!` is called.")
    end
end

include("generate.jl")
include("assess.jl")
include("propose.jl")
include("project.jl")
include("choicemap.jl")
include("update.jl")