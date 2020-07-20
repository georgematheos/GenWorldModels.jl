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

"""
    ConvertKeyAtDepthAddressTree{LT, depth, F, B} <: Gen.AddressTree{LT}

Address tree which wraps an underlying address tree, but transforms all keys at `depth`
using the given forward conversion and backward conversion functions.
- `fwd_convert(key)` converts `key` from the representation in the `tree` to the representation this
wrapped tree should expose
- `bwd_convert(key)` converts `key` from the representation the wrapped tree should expose
to the representation used in `tree`
"""
struct ConvertKeyAtDepthAddressTree{LT, depth, F, B} <: Gen.AddressTree{LT}
    tree::Gen.AddressTree{LT}
    fwd_convert::F
    bwd_convert::B
end
function ConvertKeyAtDepthAddressTree{LT, depth}(tree::Gen.AddressTree{LT}, f::F, b::B) where {LT, depth, F, B}
    ConvertKeyAtDepthAddressTree{LT, depth, F, B}(tree, f, b)
end

ConvertKeyAtDepthAddressTree{LT, depth, F, B}(tree::EmptyAddressTree, f::F, b::B) where {LT, depth, F, B} = tree
ConvertKeyAtDepthAddressTree{LT, depth, F, B}(tree::AllSelection, f::F, b::B) where {LT, depth, F, B} = tree
ConvertKeyAtDepthAddressTree{LT, depth, F, B}(tree::Value, f::F, b::B) where {LT, depth, F, B} = tree

Gen.get_subtree(t::ConvertKeyAtDepthAddressTree{<:Any, 1}, key) = get_subtree(t.tree, t.bwd_convert(key))
Gen.get_subtree(t::ConvertKeyAtDepthAddressTree{<:Any, 1}, key::Pair) = Gen._get_subtree(t, key)
function Gen.get_subtree(t::ConvertKeyAtDepthAddressTree{T, depth}, key) where {T, depth}
    ConvertKeyAtDepthAddressTree{T, depth-1}(get_subtree(t.tree, key), t.fwd_convert, t.bwd_convert)
end
Gen.get_subtree(t::ConvertKeyAtDepthAddressTree{T, depth}, key::Pair) where {T, depth} = Gen._get_subtree(t, key)

function Gen.get_subtrees_shallow(t::ConvertKeyAtDepthAddressTree{<:Any, 1})
    (
        (t.fwd_convert(key), subtree)
        for (key, subtree) in get_subtrees_shallow(t.tree)
    )
end
function Gen.get_subtrees_shallow(t::ConvertKeyAtDepthAddressTree{T, depth}) where {T, depth}
    (
        (key, ConvertKeyAtDepthAddressTree{T, depth-1}(subtree, t.fwd_convert, t.bwd_convert))
        for (key, subtree) in get_subtrees_shallow(t.tree)
    )
end

function get_to_id_and_idx(id_table::IDTable)
    to_id(key) = convert_key_to_id_form(key, id_table)
    to_idx(key) = convert_key_to_idx_form(key, id_table)
    (to_id, to_idx)
end
"""
    to_id_repr(tree, id_table)

Convert the given address tree for the world where all MGF calls use idx representation
to have all the MGF keys use id representation instead.
"""
function to_id_repr(tree::Gen.AddressTree{LT}, id_table::IDTable) where {LT}
    (f, b) = get_to_id_and_idx(id_table)
    ConvertKeyAtDepthAddressTree{LT, 2, typeof(f), typeof(b)}(tree, f, b)
end
"""
    to_idx_repr(tree, id_table)

Convert the given address tree for the world where all MGF calls use id representation
to have all the MGF keys use idx representation instead.
"""
function to_idx_repr(tree::Gen.AddressTree{LT}, id_table::IDTable) where {LT}
    (b, f) = get_to_id_and_idx(id_table)
    ConvertKeyAtDepthAddressTree{LT, 2, typeof(f), typeof(b)}(tree, f, b)
end

# if we don't have any OUPM types, don't bother wrapping the choicemap
to_id_repr(tree::Gen.AddressTree, id_table::IDTable{()}) = tree
to_idx_repr(tree::Gen.AddressTree, id_table::IDTable{()}) = tree