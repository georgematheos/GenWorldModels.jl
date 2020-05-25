###################
# World ChoiceMap #
###################

struct MemoizedGenerativeFunctionChoiceMap <: Gen.ChoiceMap
    world::World
    addr::Symbol
end

Gen.has_value(c::MemoizedGenerativeFunctionChoiceMap, addr) = false
function Gen.has_value(c::MemoizedGenerativeFunctionChoiceMap, addr::Pair)
    key, rest = addr
    call = Call(c.addr, key)
    has_value(get_choices(c.world.subtraces[call]), rest)
end

function Gen.get_value(c::MemoizedGenerativeFunctionChoiceMap, addr::Pair)
    key, rest = addr
    call = Call(c.addr, key)
    get_value(get_choices(c.world.subtraces[call]), rest)
end

function Gen.get_submap(c::MemoizedGenerativeFunctionChoiceMap, key)
    call = Call(c.addr, key)
    if has_value_for_call(c.world, call)
        get_choices(c.world.subtraces[call])
    else
        EmptyChoiceMap()
    end
end
function Gen.get_submap(c::MemoizedGenerativeFunctionChoiceMap, addr::Pair)
    key, rest = addr
    get_submap(get_submap(c, key), rest)
end

Gen.get_values_shallow(c::MemoizedGenerativeFunctionChoiceMap) = ()

# TODO: store subtraces in a way s.t. we can avoid this filter call and do a direct lookup
function Gen.get_submaps_shallow(c::MemoizedGenerativeFunctionChoiceMap)
    (key(call) => get_choices(subtrace) for (call, subtrace) in filter(((call, subtrace),) -> addr(call) == c.addr, c.world.subtraces))
end

function Base.isempty(c::MemoizedGenerativeFunctionChoiceMap)
    isempty(get_values_shallow(c)) && all(isempty(submap) for (_, submap) in (get_submaps_shallow(c)))
end

# TODO: should I use something other than a static choicemap here
# so that we can lazily construct the MemoizedGenerativeFunctionChoiceMaps
# rather than explicitly creating a tuple of them?
function Gen.get_choices(world::World{addrs, <:Any}) where {addrs}
    leaf_nodes = NamedTuple()
    internal_nodes = NamedTuple{addrs}(
        Tuple(MemoizedGenerativeFunctionChoiceMap(world, addr) for addr in addrs)
    )
    StaticChoiceMap(leaf_nodes, internal_nodes)
end