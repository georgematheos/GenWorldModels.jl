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

struct KeyChangedDiff <: Gen.Diff
    diff::Gen.Diff
end
struct ValueChangedDiff <: Gen.Diff
    diff::Gen.Diff
end
struct ToBeUpdatedDiff <: Gen.Diff end

### world[addr] diffs ###

no_addr_change_error() = error("Changing the address of a `lookup_or_generate` in updates is not supported.")
Base.getindex(world::World, addr::Diffed) = no_addr_change_error()
Base.getindex(world::World, addr::Diffed{<:Any, NoChange}) = Diffed(world[strip_diff(addr)], NoChange())
Base.getindex(world::Diffed{<:World}, addr::Diffed) = no_addr_change_error()
Base.getindex(world::Diffed{<:World}, addr::Diffed{<:Any, NoChange}) = world[strip_diff(addr)]
Base.getindex(world::Diffed{<:World, UnknownChange}, addr::CallAddr) = Diffed(strip_diff(world)[addr], UnknownChange())
Base.getindex(world::Diffed{<:World, NoChange}, addr::CallAddr) = Diffed(strip_diff(world)[addr], NoChange())

# the main one is: a diffed world --> a diffed mgf with a WorldUpdateDiff
Base.getindex(world::Diffed{<:World, WorldUpdateDiff}, addr::CallAddr) = Diffed(strip_diff(world)[addr], WorldUpdateDiff())

### MGFCall diff propagation ###

Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, WorldUpdateDiff}, key::Diffed{<:Any, NoChange}) = mgf[strip_diff(key)]
Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, UnknownChange}, key) = Diffed(strip_diff(mgf)[strip_diff(key)], UnknownChange())
Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, NoChange}, key) = Diffed(strip_diff(mgf)[key], NoChange())

# keychange
function Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, WorldUpdateDiff}, key::Diffed)
    Diffed(strip_diff(mgf)[strip_diff(key)], KeyChangedDiff(get_diff(key)))
end

# valchange, tobeupdated, or nochange
function Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, WorldUpdateDiff}, key)
    mgf = strip_diff(mgf)
    wrld = world(mgf)
    c = Call(addr(mgf), key)

    if haskey(wrld.state.diffs, c)
        Diffed(mgf[key], ValueChangedDiff(wrld.state.diffs[c]))
    else
        if already_updated(wrld, c)
            Diffed(mgf[key], NoChange())
        else
            Diffed(mgf[key], ToBeUpdatedDiff())
        end
    end
end
Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction, WorldUpdateDiff}, key::ConcreteIndexOUPMObject) = Diffed(strip_diff(mgf)[key], ToBeUpdatedDiff())

# get abstract form!
function Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction{<:Any, _get_abstract_addr}, WorldUpdateDiff}, key::ConcreteIndexOUPMObject)
    mgf = strip_diff(mgf)
    wrld = world(mgf)
    c = Call(addr(mgf), key)
    if haskey(wrld.state.diffs, c)
        Diffed(mgf[key], ValueChangedDiff(wrld.state.diffs[c]))
    else
        if has_val(wrld, c)
            Diffed(mgf[key], NoChange())
        else
            Diffed(mgf[key], ToBeUpdatedDiff())
        end
    end
end
# to avoid ambiguity we need to define:
function Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction{<:Any, _get_abstract_addr}, WorldUpdateDiff}, key::Diffed)
    Diffed(strip_diff(mgf)[strip_diff(key)], KeyChangedDiff(get_diff(key)))
end
Base.getindex(mgf::Diffed{<:MemoizedGenerativeFunction{<:Any, _get_abstract_addr}, WorldUpdateDiff}, key::Diffed{<:Any, NoChange}) = mgf[strip_diff(key)]

###################################################
# Diffs for `world`, `key`, and `addr` of MGFCall #
###################################################
# world object always has a world update diff for any of the custom update types
function world(call::Diffed{<:MemoizedGenerativeFunctionCall, <:Union{ToBeUpdatedDiff, KeyChangedDiff, ValueChangedDiff}})
    wrld = world(strip_diff(call))
    Diffed(wrld, WorldUpdateDiff())
end
function world(call::Diffed{<:MemoizedGenerativeFunctionCall, NoChange})
    wrld = world(strip_diff(call))
    Diffed(wrld, NoChange())
end

key(call::Diffed{<:MemoizedGenerativeFunctionCall, NoChange}) = Diffed(key(strip_diff(call)), NoChange())
key(call::Diffed{<:MemoizedGenerativeFunctionCall, ValueChangedDiff}) = Diffed(key(strip_diff(call)), NoChange())
key(call::Diffed{<:MemoizedGenerativeFunctionCall, ToBeUpdatedDiff}) = Diffed(key(strip_diff(call)), NoChange())
key(call::Diffed{<:MemoizedGenerativeFunctionCall, KeyChangedDiff}) = Diffed(key(strip_diff(call)), get_diff(call).diff)

# if diffed with a different diff, we have to put UnknownChange for the key
key(call::Diffed{<:MemoizedGenerativeFunctionCall}) = Diffed(key(strip_diff(call)), UnknownChange())

# for all the custom diff types, there is no change to the address of the MGF call
function addr(call::Diffed{<:MemoizedGenerativeFunctionCall, <:Union{
    NoChange, KeyChangedDiff, ToBeUpdatedDiff, ValueChangedDiff
}})
    Diffed(addr(strip_diff(call)), NoChange())
end