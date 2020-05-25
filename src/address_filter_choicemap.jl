"""
    AddressFilterChoicemap(choicemap::ChoiceMap, f::Function)

Wraps `choicemap` and filters out branches with a top-level address such that
`f(address)` is false.

Ie. this exposes all choices in `choicemap`, unless the address `node1 => node2 => ... => node_n`
is such that `f(node_m)` is false, in which case this choice will not appear in the choicemap.
"""
struct AddressFilterChoicemap <: Gen.ChoiceMap
    choicemap::ChoiceMap
    f::Function
end

Gen.has_value(c::AddressFilterChoicemap, addr) = c.f(addr) && has_value(c.choicemap, addr)
function Gen.has_value(c::AddressFilterChoicemap, addr::Pair)
    (first, rest) = addr
    has_value(get_submap(c, first), rest)
end

function Gen.get_submap(c::AddressFilterChoicemap, addr)
    if !c.f(addr)
        EmptyChoiceMap()
    else
        AddressFilterChoicemap(get_submap(c.choicemap, addr), c.f)
    end
end
function Gen.get_submap(c::AddressFilterChoicemap, addr::Pair)
    first, rest = addr
    if !c.f(first)
        EmptyChoiceMap()
    else
        submap = get_submap(c, first)
        get_submap(submap, rest)
    end
end

function Gen.get_value(c::AddressFilterChoicemap, addr)
    if !c.f(addr)
        throw(KeyError(addr))
    else
        get_value(c.choicemap, addr)
    end
end
function Gen.get_value(c::AddressFilterChoicemap, addr::Pair)
    first, rest = addr
    if !c.f(first)
        throw(KeyError(first))
    else
        get_value(get_submap(c, first), rest)
    end
end

function Gen.get_values_shallow(c::AddressFilterChoicemap)
    (
        addr_val
        for addr_val in get_values_shallow(c.choicemap)
        if c.f(addr_val[1])
    )
end

function Gen.get_submaps_shallow(c::AddressFilterChoicemap)
    (
        (addr, AddressFilterChoicemap(submap, c.f))
        for (addr, submap) in get_submaps_shallow(c.choicemap)
        if c.f(addr)
    )
end

function Base.isempty(c::AddressFilterChoicemap)
    isempty(get_values_shallow(c)) && all(isempty(submap) for (_, submap) in (get_submaps_shallow(c)))
end