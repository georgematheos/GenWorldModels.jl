###################
# World ChoiceMap #
###################

struct MemoizedGenerativeFunctionChoiceMap <: Gen.AddressTree{Gen.Value}
    world::World
    addr::Symbol
end

function Gen.get_subtree(c::MemoizedGenerativeFunctionChoiceMap, key)
    call = Call(c.addr, key)
    if has_value_for_call(c.world, call)
        get_choices(c.world.subtraces[call])
    else
        EmptyChoiceMap()
    end
end
Gen.get_subtree(c::MemoizedGenerativeFunctionChoiceMap, addr::Pair) = Gen._get_subtree(c, addr)
Gen.get_values_shallow(c::MemoizedGenerativeFunctionChoiceMap) = ()

# TODO: store subtraces in a way s.t. we can avoid this filter call and do a direct lookup
function Gen.get_subtrees_shallow(c::MemoizedGenerativeFunctionChoiceMap)
    (
        key(call) => get_choices(subtrace)
        for (call, subtrace) in
            filter(((call, subtrace),) -> addr(call) == c.addr, c.world.subtraces)
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