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
struct MemoizedGenerativeFunctionCall{WorldType, addr, KeyType}
    world::WorldType
    key::KeyType
end
MemoizedGenerativeFunctionCall(world::WorldType, addr::CallAddr, key::KeyType) where {WorldType, KeyType} = MemoizedGenerativeFunctionCall{WorldType, addr, KeyType}(world, key)
addr(::MemoizedGenerativeFunctionCall{<:Any, a}) where {a} = a
key(mgf::MemoizedGenerativeFunctionCall) = mgf.key
call(mgf::MemoizedGenerativeFunctionCall) = Call(addr(mgf), key(mgf))
world(mgf::MemoizedGenerativeFunctionCall) = mgf.world

# world[:addr] gives a memoized gen function
# world[:addr][key] gives a memoized gen function call
Base.getindex(world::World, addr::CallAddr) = MemoizedGenerativeFunction(world, addr)
Base.getindex(mgf::MemoizedGenerativeFunction, key) = MemoizedGenerativeFunctionCall(world(mgf), addr(mgf), key)

###########################################
# Argdiff propagation for MGF and MGFCall #
###########################################

### world[addr] diffs ###

no_addr_change_error() = error("Changing the address of a `lookup_or_generate` in updates is not supported.")
Base.getindex(world::World, addr::Diffed) = no_addr_change_error()
Base.getindex(world::World, addr::Diffed{<:Any, NoChange}) = Diffed(world[strip_diff(addr)], NoChange())
Base.getindex(world::Diffed{<:World}, addr::Diffed) = no_addr_change_error()
Base.getindex(world::Diffed{<:World}, addr::Diffed{<:Any, NoChange}) = world[strip_diff(addr)]

function Base.getindex(world::Diffed{<:World, WorldUpdateDiff}, addr::CallAddr)
    mgf = strip_diff(world)[addr]
    diff = get_diff(world)[addr] # will be a WorldUpdateAddrDiff
    Diffed(mgf, diff)
end
Base.getindex(world::Diffed{<:World, UnknownChange}, addr::CallAddr) = Diffed(strip_diff(world)[addr], UnknownChange())
Base.getindex(world::Diffed{<:World, NoChange}, addr::CallAddr) = Diffed(strip_diff(world)[addr], NoChange())

### SIMPLE MGFCall diff propagation (where no idx --> id conversion is needed) ###

"""
    MGFCallKeyChangeDiff

Denotes that a memoized gen function call has had its key changed.
(The MGF call may or may not have also have had its value changed.)
"""
struct MGFCallKeyChangeDiff <: Gen.Diff
    key_diff::Gen.Diff
end

"""
    MGFCallValChangeDiff

Denotes that a memoized gen function call which has not had its key change
has had its value change (due to a change in the value for this call in the world
associated with the MGFCall).
"""
struct MGFCallValChangeDiff <: Gen.Diff
    diff::Diff
end

function Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, WorldUpdateAddrDiff}, key)
    mgf_call = strip_diff(mgf)[key]
    diff = get_diff(mgf)[key] # may be a nochange, if !has_diff_for_index(get_diff(mgf), key)
    # unless this is either NoChange or ToBeUpdated, we want to make sure `update` knows that this is just
    # a normal diff which comes from a value change, and doesn't need to trigger special update behavior
    if diff !== NoChange() && diff !== ToBeUpdatedDiff()# && diff !== IDAssociationChanged()
        diff = MGFCallValChangeDiff(diff)
    end

    Diffed(mgf_call, diff)
end

function Base.getindex(mgf::MemoizedGenerativeFunction, key::Diffed)
    key_diff = get_diff(key)
    key = strip_diff(key)
    mgf_call = mgf[key]
    Diffed(mgf_call, MGFCallKeyChangeDiff(key_diff))
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
        Diffed(mgf_call, MGFCallKeyChangeDiff(get_diff(key)))
    end
end

Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, UnknownChange}, key) = Diffed(strip_diff(mgf)[strip_diff(key)], UnknownChange())
Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, NoChange}, key) = Diffed(strip_diff(mgf)[key], NoChange())

### Overwrite this method for the case where we need idx --> id conversion so we can check if the associated id has changed ###
struct WorldDiffedNoKeyChange <: Gen.Diff end
function Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, WorldUpdateAddrDiff}, key::ConcreteIndexOUPMObject)
    Diffed(strip_diff(mgf)[key], WorldDiffedNoKeyChange())
end

###################################################
# Diffs for `world`, `key`, and `addr` of MGFCall #
###################################################
# world object always has a world update diff
function world(call::Diffed{<:MemoizedGenerativeFunctionCall})
    wrld = world(strip_diff(call))
    Diffed(wrld, WorldUpdateDiff(wrld))
end
function world(call::Diffed{<:MemoizedGenerativeFunctionCall, NoChange})
    wrld = world(strip_diff(call))
    Diffed(wrld, NoChange())
end

# there are several mgf_call diffs which denote there is no keychange
key(call::Diffed{<:MemoizedGenerativeFunctionCall, WorldDiffedNoKeyChange}) = Diffed(key(strip_diff(call)), NoChange())
key(call::Diffed{<:MemoizedGenerativeFunctionCall, NoChange}) = Diffed(key(strip_diff(call)), NoChange())
key(call::Diffed{<:MemoizedGenerativeFunctionCall, MGFCallValChangeDiff}) = Diffed(key(strip_diff(call)), NoChange())
key(call::Diffed{<:MemoizedGenerativeFunctionCall, ToBeUpdatedDiff}) = Diffed(key(strip_diff(call)), NoChange())
key(call::Diffed{<:MemoizedGenerativeFunctionCall, MGFCallKeyChangeDiff}) = Diffed(key(strip_diff(call)), get_diff(call).key_diff)

# if diffed with a different diff, we have to put UnknownChange for the key
key(call::Diffed{<:MemoizedGenerativeFunctionCall}) = Diffed(key(strip_diff(call)), UnknownChange())

# for all the custom diff types, they communicate that the address has not changed.
function addr(call::Diffed{<:MemoizedGenerativeFunctionCall, <:Union{
    WorldDiffedNoKeyChange, NoChange, MGFCallKeyChangeDiff, ToBeUpdatedDiff, MGFCallKeyChangeDiff, WorldDiffedNoKeyChange
}})
    Diffed(addr(strip_diff(call)), NoChange())
end