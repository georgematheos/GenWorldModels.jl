"""
    AddressFilterChoiceMap(choicemap::ChoiceMap, f::Function)

Wraps `choicemap` and filters out branches with a top-level address such that
`f(address)` is false.

Ie. this exposes all choices in `choicemap`, unless the address `node1 => node2 => ... => node_n`
is such that `f(node_m)` is false, in which case this choice will not appear in the choicemap.
"""
struct AddressFilterAddressTree{LeafType} <: Gen.AddressTree{LeafType}
    tree::Gen.AddressTree{LeafType}
    f::Function
end
const AddressFilterChoiceMap = AddressFilterAddressTree{Gen.Value}
AddressFilterAddressTree(t::Gen.AddressTreeLeaf, f::Function) = t
AddressFilterAddressTree{LT}(t::Gen.AddressTreeLeaf, f::Function) where {LT} = t

function Gen.get_subtree(t::AddressFilterAddressTree, addr)
    if !t.f(addr)
        EmptyAddressTree()
    else
        AddressFilterAddressTree(get_subtree(t.tree, addr), t.f)
    end
end
Gen.get_subtree(t::AddressFilterAddressTree, addr::Pair) = Gen._get_subtree(t, addr)

function Gen.get_subtrees_shallow(t::AddressFilterAddressTree)
    ( # TODO: should we filter out the empty submaps using `Gen.nonempty_subtree_itr`?
        (addr, AddressFilterAddressTree(subtree, t.f))
        for (addr, subtree) in get_subtrees_shallow(t.tree)
        if t.f(addr)
    )
end
