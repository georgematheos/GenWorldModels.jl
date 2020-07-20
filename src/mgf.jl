"""
    mgf.jl

Code for `MemoizedGenerativeFunction` and `MemoizedGenerativeFunctionCall`
types.

`world[address]` yields a MGF;
`world[address][key]` yields a MGFCall.
"""

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
MemoizedGenerativeFunction(world::WorldType, addr::CallAddr) where {WorldType} = MemoizedGenerativeFunction{WorldType, addr}(world)
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
MemoizedGenerativeFunctionCall(world::WorldType, addr::CallAddr, key) where {WorldType} = MemoizedGenerativeFunctionCall{WorldType, addr}(world, key)
addr(::MemoizedGenerativeFunctionCall{<:Any, a}) where {a} = a
key(mgf::MemoizedGenerativeFunctionCall) = mgf.key
call(mgf::MemoizedGenerativeFunctionCall) = Call(addr(mgf), key(mgf))

# world[:addr] gives a memoized gen function
# world[:addr][key] gives a memoized gen function call
Base.getindex(world::World, addr::CallAddr) = MemoizedGenerativeFunction(world, addr)
Base.getindex(mgf::MemoizedGenerativeFunction, key) = MemoizedGenerativeFunctionCall(world(mgf), addr(mgf), key)

# if the key is an OUPM object in index form, we should automatically convert
# to identifier form
function Base.getindex(mgf::MemoizedGenerativeFunction, key::OUPMType{Int})
    T = oupm_type(key)
    idx = key.idx_or_id
    get_id_call = Call(T, idx)
    if !has_val(mgf.world, get_id_call)
        generate_id_for_call!(mgf.world, get_id_call)
    end
    id_obj = get_val(mgf.world, get_id_call)
    return mgf[id_obj]
end

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
Base.getindex(world::Diffed{<:World, UnknownChange}, addr::CallAddr) = Diffed(strip_diff(world)[addr], UnknownChange())
Base.getindex(world::Diffed{<:World, NoChange}, addr::CallAddr) = Diffed(strip_diff(world)[addr], NoChange())

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
