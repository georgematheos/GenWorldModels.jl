####################################
# LookupOrGenerate Parameter Types #
####################################

# TODO: should I include more information in the type for these?
# also TODO: have world[addr] syntax use `Val` dispatch so it happens at compile-time

"""
    MemoizedGenerativeFunction

Represents a memoized generative function along with the world
the generative function is memoized in the context of.

# Eg.
```julia
memoized_generative_function = world[:mgf_address]
```
"""
struct MemoizedGenerativeFunction{WorldType, addr}
    world::WorldType
end
MemoizedGenerativeFunction(world::WorldType, addr::Symbol) where {WorldType} = MemoizedGenerativeFunction{WorldType, addr}(world)
addr(mgf::MemoizedGenerativeFunction{<:Any, a}) where {a} = a
world(mgf::MemoizedGenerativeFunction) = mgf.world

"""
    MemoizedGenerativeFunctionCall

Represents a call to a memoized generative function in a world,
along with the world the generative function is memoized in the context of.

# Eg.
```julia
memoized_generative_function = world[:mgf_address]
memoized_generative_function_call = memoized_generative_function[argument_to_memoized_generative_function]
```

"""
struct MemoizedGenerativeFunctionCall{WorldType, addr}
    world::WorldType
    key
end
MemoizedGenerativeFunctionCall(world::WorldType, addr::Symbol, key) where {WorldType} = MemoizedGenerativeFunctionCall{WorldType, addr}(world, key)
addr(::MemoizedGenerativeFunctionCall{<:Any, a}) where {a} = a
key(mgf::MemoizedGenerativeFunctionCall) = mgf.key
call(mgf::MemoizedGenerativeFunctionCall) = Call(addr(mgf), key(mgf))

# world[:addr] gives a memoized gen function
# world[:addr][key] gives a memoized gen function call
Base.getindex(world::World, addr::Symbol) = MemoizedGenerativeFunction(world, addr)
Base.getindex(mgf::MemoizedGenerativeFunction, key) = MemoizedGenerativeFunctionCall(world(mgf), addr(mgf), key)

###########################################
# Argdiff propagation for MGF and MGFCall #
###########################################

"""
    MGFCallKeyChangeDiff

Denotes that a memoized gen function call has had its key changed.
(The MGF call may or may not have also have had its value changed.)
"""
struct MGFCallKeyChangeDiff <: Gen.Diff end

"""
    MGFCallValChangeDiff

Denotes that a memoized gen function call which has not had its key change
has had its value change (due to a change in the value for this call in the world
associated with the MGFCall).
"""
struct MGFCallValChangeDiff <: Gen.Diff
    diff::Diff
end

no_addr_change_error() = error("Changing the address of a `lookup_or_generate` in updates is not supported.")
Base.getindex(world::World, addr::Diffed) = no_addr_change_error()
Base.getindex(world::Diffed{<:World}, addr::Diffed) = no_addr_change_error()

function Base.getindex(world::Diffed{<:World, WorldUpdateDiff}, addr::Symbol)
    mgf = strip_diff(world)[addr]
    diff = get_diff(world)[addr] # will be a WorldUpdateAddrDiff
    Diffed(mgf, diff)
end
Base.getindex(world::Diffed{<:World, UnknownChange}, addr::Symbol) = Diffed(strip_diff(world)[addr], UnknownChange())
Base.getindex(world::Diffed{<:World, NoChange}, addr::Symbol) = Diffed(strip_diff(world)[addr], NoChange())

function Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, WorldUpdateAddrDiff}, key)
    mgf_call = strip_diff(mgf)[key]
    diff = get_diff(mgf)[key] # may be a nochange, if !has_diff_for_index(get_diff(mgf), key)
    # unless this is either NoChange or ToBeUpdated, we want to make sure `update` knows that this is just
    # a normal diff which comes from a value change, and doesn't need to trigger special update behavior
    if diff != NoChange() && diff != ToBeUpdatedDiff()
        diff = MGFCallValChangeDiff(diff)
    end

    Diffed(mgf_call, diff)
end

function Base.getindex(mgf::MemoizedGenerativeFunction, key::Diffed)
    key = strip_diff(key)
    mgf_call = mgf[key]
    Diffed(mgf_call, MGFCallKeyChangeDiff())
end

# key diff supercedes a diff on the world because we ALWAYS need to run an update on the world
# to communicate a dependency change when the key changes, while we sometimes don't run a dependency
# change update when there's only a change to the world
function Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, WorldUpdateAddrDiff}, key::Diffed)
    mgf_call = strip_diff(mgf)[strip_diff(key)]
    if get_diff(key) == NoChange()
        # if there's no diff on the key, resort to our getindex for undiffed keys
        mgf[strip_diff(key)]
    else
        Diffed(mgf_call, MGFCallKeyChangeDiff())
    end
end
Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, UnknownChange}, key) = Diffed(strip_diff(mgf)[strip_diff(key)], UnknownChange())
Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, NoChange}, key) = strip_diff(mgf)[key]

####################
# LookupOrGenerate #
####################

# TODO: could add some more information about the return type to the trace

struct LookupOrGenerateTrace <: Gen.Trace
    call::MemoizedGenerativeFunctionCall
    val
end

Gen.get_args(tr::LookupOrGenerateTrace) = (tr.call,)
Gen.get_retval(tr::LookupOrGenerateTrace) = tr.val
Gen.get_score(tr::LookupOrGenerateTrace) = 0.
Gen.get_gen_fn(tr::LookupOrGenerateTrace) = lookup_or_generate
Gen.project(tr::LookupOrGenerateTrace, selection::EmptySelection) = 0.
function Gen.get_choices(tr::LookupOrGenerateTrace)
    # TODO: static choicemap for performance?
    choicemap(
        (metadata_addr(tr.call.world) => addr(tr.call), key(tr.call))
    )
end

struct LookupOrGenerate <: GenerativeFunction{Any, LookupOrGenerateTrace} end
const lookup_or_generate = LookupOrGenerate()

@inline (gen_fn::LookupOrGenerate)(args...) = get_retval(simulate(gen_fn, args))
@inline Gen.simulate(gen_fn::LookupOrGenerate, args::Tuple) = Gen.generate(gen_fn, args)[1]

function Gen.generate(gen_fn::LookupOrGenerate, args::Tuple{MemoizedGenerativeFunctionCall{WorldType, addr}}, constraints::EmptyChoiceMap) where {WorldType, addr}
    mgf_call, = args
    val = lookup_or_generate!(mgf_call.world, Call(addr, mgf_call.key); reason_for_call=:generate)
    tr = LookupOrGenerateTrace(mgf_call, val)
    (tr, 0.)
end

function Gen.generate(gen_fn::LookupOrGenerate, args::Tuple, constraints::ChoiceMap)
    error("generate(lookup_or_generate, ...) should only be called with empty constraints")
end

function Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap)
    error("lookup_or_generate may not be updated with constraints.")
end

Gen.update(tr::LookupOrGenerateTrace, ::Tuple, ::Tuple{NoChange}, ::EmptyChoiceMap) = (tr, 0., NoChange(), EmptyChoiceMap())

# key change
function Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{MGFCallKeyChangeDiff}, ::EmptyChoiceMap)
    mgf_call = args[1]

    # run a full update/generate cycle in the world for this call
    new_val = lookup_or_generate!(mgf_call.world, Call(addr(mgf_call), key(mgf_call)); reason_for_call=:key_change)
    
    # the key looked up is the only thing exposed in the choicemap; this has changed so the whole old choicemap is the discard
    discard = get_choices(tr)
    
    new_tr = LookupOrGenerateTrace(mgf_call, new_val)
    retdiff = new_val == get_retval(tr) ? NoChange() : UnknownChange()
    (new_tr, 0., retdiff, discard)
end

# to-be-updated
function Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{ToBeUpdatedDiff}, ::EmptyChoiceMap)
    mgf_call = args[1]
    # run a full update/generate cycle in the world for this call
    new_val = lookup_or_generate!(mgf_call.world, Call(addr(mgf_call), key(mgf_call)); reason_for_call=:to_be_updated)
    new_tr = LookupOrGenerateTrace(mgf_call, new_val)
    retdiff = new_val == get_retval(tr) ? NoChange() : UnknownChange()
    (new_tr, 0., retdiff, EmptyChoiceMap())
end

# value has changed, but we don't need to update, and haven't changed key
function Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{MGFCallValChangeDiff}, ::EmptyChoiceMap)
    valdiff = argdiffs[1].diff
    new_call = args[1]
    new_val = get_value_for_call(new_call.world, call(new_call))
    new_tr = LookupOrGenerateTrace(new_call, new_val)
    (new_tr, 0., valdiff, EmptyChoiceMap())
end

# UnknownChange - this is likely caused by use of a dynamic generative function which doesn't propagate diffs
# we need to figure out which of the other update functions we should dispatch to
# this function just replicates the diff propagation behavior implemented above in the Base.getindex methods
function Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{UnknownChange}, ::EmptyChoiceMap)
    new_call = args[1]
    if key(tr.call) != key(new_call)
        diff = MGFCallKeyChangeDiff()
    else
        diff = WorldUpdateDiff(new_call.world)[addr(new_call)][key(new_call)]
        if diff != NoChange() && diff != ToBeUpdatedDiff()
            diff = MGFCallValChangeDiff(diff)
        end
    end
    return Gen.update(tr, args, (diff,), EmptyChoiceMap())
end

# TODO: regenerate

# TODO: gradients