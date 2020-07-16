###################
# World ChoiceMap #
###################

struct MemoizedGenerativeFunctionChoiceMap <: Gen.AddressTree{Gen.Value}
    world::World
    addr::Symbol
end

function Gen.get_subtree(c::MemoizedGenerativeFunctionChoiceMap, key)
    call = Call(c.addr, key)
    if has_val(c.world, call)
        get_choices(get_trace(c.world, call))
    else
        EmptyChoiceMap()
    end
end
Gen.get_subtree(c::MemoizedGenerativeFunctionChoiceMap, addr::Pair) = Gen._get_subtree(c, addr)
Gen.get_values_shallow(c::MemoizedGenerativeFunctionChoiceMap) = ()

# TODO: store subtraces in a way s.t. we can avoid this filter call and do a direct lookup
function Gen.get_subtrees_shallow(c::MemoizedGenerativeFunctionChoiceMap)
    (
        lookup_key => get_choices(subtrace)
        for (lookup_key, subtrace) in traces_for_mgf(c.world.traces, c.addr)
    )
end

# TODO: should I use something other than a static choicemap here
# so that we can lazily construct the MemoizedGenerativeFunctionChoiceMaps
# rather than explicitly creating a tuple of them?
function Gen.get_choices(world::World{addrs, <:Any}) where {addrs}
    StaticChoiceMap(NamedTuple{addrs}(
        Tuple(MemoizedGenerativeFunctionChoiceMap(world, addr) for addr in addrs)
    ))
end