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

# special addresses for world args and getting the index of a OUPM object
const _world_args_addr = :args
const _get_index_addr = :index
const _get_origin_addr = :origin
const _get_abstract_addr = :abstract
const _get_concrete_addr = :concrete
const _special_addrs = (_world_args_addr, _get_index_addr, _get_origin_addr, _get_abstract_addr, _get_concrete_addr)
const _NonMGFCall = Union{(Call{a} for a in _special_addrs)...}
is_mgf_call(c::Call{_world_args_addr}) = false
is_mgf_call(c::Call{_get_index_addr}) = false
is_mgf_call(c::Call{_get_origin_addr}) = false
is_mgf_call(c::Call{_get_abstract_addr}) = false
is_mgf_call(c::Call{_get_concrete_addr}) = false
is_mgf_call(c::Call) = true

# data structures
include("data_structures/traces.jl") # `Traces` data structure to store subtraces
include("data_structures/lookup_counts.jl") # `LookupCounts` data structure to store the dependency graph & the counts
include("data_structures/call_sort.jl") # `CallSort` data structure to maintain a topological numbering of the calls.
include("data_structures/id_table.jl") # Association between IDs and indices for OUPM types

"""
    World
    
A `World` is a context in which generative functions can be memoized;
it stores the values which have been generated for a memoized generative functions
for each argument it has been called on.
    
Every memoized generative function for a given world has a specific address which
is used to refer to it.
"""
mutable struct World{addrs, GenFnTypes, WorldArgnames}
    gen_fns::GenFnTypes # tuple of the generative functions for this world
    state::WorldState # a state representing what operation is currently being performed on the world (eg. update, generate)
    traces::Traces
    world_args::NamedTuple{WorldArgnames}
    lookup_counts::LookupCounts # tracks how many times each call was performed, and which calls were performed from which
    call_sort::CallSort # a topological sort of which calls must be finished before the next can begin
    id_table::IDTable # associations between IDs and objects
    total_score::Float64
    metadata_addr::Symbol # an address for `lookup_or_generate`s from this world in choicemaps
end

function World{addrs, GenFnTypes, WorldArgnames}(gen_fns, state, traces, world_args, lookup_counts, call_sort, id_table, total_score) where {addrs, GenFnTypes, WorldArgnames}
    World{addrs, GenFnTypes, WorldArgnames}(gen_fns, state, traces, world_args, lookup_counts, call_sort, id_table, total_score, gensym("WORLD_LOOKUP"))
end

World(w::World{A,G,N}) where {A,G,N} = World{A,G,N}(w.gen_fns, w.state, w.traces, w.world_args, w.lookup_counts, w.call_sort, w.id_table, w.total_score, w.metadata_addr)

function World{addrs, GenFnTypes, WorldArgnames}(gen_fns, world_args::NamedTuple{WorldArgnames}) where {addrs, GenFnTypes, WorldArgnames}
    w = World{addrs, GenFnTypes, WorldArgnames}(
        gen_fns,
        NoChangeWorldState(),
        Traces(addrs, gen_fns),
        world_args,
        LookupCounts(),
        CallSort(),
        IDTable(),
        0.
    )
    # note a call for every world arg
    for (arg_address, _) in pairs(world_args)
        note_new_call!(w, Call(_world_args_addr, arg_address))
    end
    return w
end

function World(addrs::NTuple{n, Symbol}, gen_fns::NTuple{n, Gen.GenerativeFunction}, world_args::NamedTuple{Names}) where {n, Names}
    for a in addrs
        @assert !(a in _special_addrs) "Cannot use reserved address $a as an MGF address."
    end

    World{addrs, typeof(gen_fns), Names}(gen_fns, world_args)
end

# for testing:
blank_world() = World((), (), NamedTuple())

# TODO: Could make this more informative by showing what calls it has in it
function Base.show(io::IO, ::Type{<:World{addrs}}) where {addrs}
    print(io, "World{")
    print(io, addrs)
    print(io, "}")
end
function Base.show(io::IO, world::World)
    print(io, typeof(world))
    print(io, "()")
end

metadata_addr(world::World) = world.metadata_addr

# NOTE: this should never be called from within a generative function; these MUST use calls to `lookup_or_generate`
_world_args(world::World) = world.world_args

# functions for tracking counts and dependency structure
"""
    note_new_call!(world, call)

Tells the dependency tracker to create table entries for a new call `call`.

Upon `has_val(world, call)` becoming true, this function (or `note_new_call_if_none_exists!`)
must be called immediately to maintain the invariant that if `has_val(world, call)`, then
the dependency tracker has table entries for it.

This will throw an error if there is already a dependency tracker entry for `call`.
"""
function note_new_call!(world, args...)
    world.lookup_counts = note_new_call(world.lookup_counts, args...)
end
"""
    note_new_call_if_none_exists!(world, call)

Like `note_new_call!`, but has no effect if the dependency tracker already has an entry for `call`
(whereas `note_new_call!` will throw an error in this case).
"""
function note_new_call_if_none_exists!(world, args...)
    world.lookup_counts = note_new_call_if_none_exists(world.lookup_counts, args...)
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

function cannot_change_retval_due_to_diffs(world::W, addr::CallAddr, argtype) where {W}
    gen_fn = get_gen_fn(world, addr)
    # return needs_nonempty_spec_for_output_change(tracetype)
    # return can_statically_guarantee_nochange_on_update(tracetype, Tuple{WorldUpdateDiff, NoChange}, EmptyAddressTree)
    rettype = Core.Compiler.return_type(
        Gen.update,
        Tuple{Gen.get_trace_type(gen_fn), Tuple{W, argtype}, Tuple{WorldUpdateDiff, NoChange}, EmptyAddressTree, AllSelection}
    )
    retdiff_statically_known = hasproperty(rettype, :parameters) && length(rettype.parameters) >= 3
    return retdiff_statically_known && rettype.parameters[3] == NoChange
end

@inline get_all_calls_which_look_up(world::World, call) = get_all_calls_which_look_up(world.lookup_counts, call)
@inline get_trace(world::World, call::Call) = get_trace(world.traces, call)
@inline total_score(world::World) = world.total_score

@generated function get_val(world::World, call::Call{call_addr}) where {call_addr}
    if call_addr == _world_args_addr
        quote world.world_args[key(call)] end
    elseif call_addr == _get_index_addr
        quote
            if key(call) isa ConcreteIndexOUPMObject
                key(call).idx
            else
                world.id_table[key(call)].idx
            end
        end
    elseif call_addr == _get_origin_addr
        quote
            if key(call) isa ConcreteIndexOUPMObject
                key(call).origin
            else
                world.id_table[key(call)].origin
            end
        end
    elseif call_addr === _get_abstract_addr
        quote
            key(call) isa AbstractOUPMObject ? key(call) : world.id_table[key(call)]  
        end
    elseif call_addr === _get_concrete_addr
        quote
            key(call) isa ConcreteIndexOUPMObject ? key(call) : world.id_table[key(call)]
        end
    else # is mgf call
        quote get_retval(get_trace(world, call)) end
    end
end
@generated function has_val(world::World, call::Call{call_addr}) where {call_addr}
    if call_addr == _world_args_addr
        quote haskey(world.world_args, key(call)) end
    elseif call_addr in (_get_index_addr, _get_origin_addr, _get_abstract_addr, _get_concrete_addr)
        quote haskey(world.id_table, key(call)) end
    else
        quote has_trace(world.traces, call) end
    end
end

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
    @assert !has_val(world, call)

    gen_fn = get_gen_fn(world, call)
    tr, weight = generate_fn(world.state)(gen_fn, (world, key(call)), constraints)
    world.traces = assoc(world.traces, call, tr)
    note_new_call!(world, call)
    world.total_score += get_score(tr)

    return weight
end
function generate_value!(world, call::Call{_get_abstract_addr}, ::EmptyAddressTree)
    @assert !(world.state isa NoChangeWorldState)
    @assert !has_val(world, call)
    generate_abstract_object!(world, key(call))
    return 0.
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
function lookup_or_generate!(world::World, call::Call, is_new_lookup)    
    if world.state isa UpdateWorldState
        lookup_or_generate_during_update!(world, call, is_new_lookup)
    elseif world.state isa GenerateWorldState
        lookup_or_generate_during_generate!(world, call)
    else
        error("World must be in a Generate or Update state if `lookup_or_generate!` is called.")
    end
end

include("id_interface.jl") # wrappers for conversion abstract <--> concrete; methods to generate abstract objects
include("gfi/generate.jl")
include("gfi/assess.jl")
include("gfi/propose.jl")
include("gfi/project.jl")
include("choicemap.jl")
include("gfi/update.jl")