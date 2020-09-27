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

Gen.get_address_schema(t::Type{<:AddressFilterAddressTree}) = DynamicAddressSchema()

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
    tree::Gen.AddressTree{<:LT}
    fwd_convert::F
    bwd_convert::B
end
function ConvertKeyAtDepthAddressTree{LT, depth}(tree, f::F, b::B) where {LT, depth, F, B}
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

"""
    to_abstract_repr(world, tree)

Convert the given address tree for the world where all MGF calls use the concrete representation
to have all the MGF keys use abstract representation instead.
"""
function to_abstract_repr(world::World, tree::Gen.AddressTree{LT}; depth=2) where {LT}
    to_abst(key) = convert_to_abstract(world, key)
    to_conc(key) = convert_to_concrete(world, key)
    ConvertKeyAtDepthAddressTree{LT, depth, typeof(to_abst), typeof(to_conc)}(tree, to_abst, to_conc)
end

"""
    to_concrete_repr(world, tree)

Convert the given address tree for the world where all MGF calls use abstract representation
to have all the MGF keys use concrete representation instead.
"""
function to_concrete_repr(world::World, tree::Gen.AddressTree{LT}; depth=2) where {LT}
    to_abst(key) = convert_to_abstract(world, key)
    to_conc(key) = convert_to_concrete(world, key)
    ConvertKeyAtDepthAddressTree{LT, depth, typeof(to_conc), typeof(to_abst)}(tree, to_conc, to_abst)
end

"""
    to_abstract_repr!(world, tree)

Convert the given address tree for the world where all MGF calls use concrete representation
to have all the MGF keys use id representation instead; lazily generate an abstract form in the world for any
concrete object for which there is currently no abstract form in the world.
"""
function to_abstract_repr!(world::World, tree::Gen.AddressTree{LT}; depth=2) where {LT}
    to_abst!(key) = convert_to_abstract!(world, key)
    to_conc(key) = convert_to_concrete(world, key)
    ConvertKeyAtDepthAddressTree{LT, depth, typeof(to_abst!), typeof(to_conc)}(tree, to_abst!, to_conc)
end

"""
    ConvertValueChoiceMap(choicemap::ChoiceMap, convert::Function)

Returns a choicemap which lazily applies the function `convert`
to every value of the given choicemap.
"""
struct ConvertValueChoiceMap{C} <: Gen.AddressTree{Value}
    cm::C
    convert::Function
    ConvertValueChoiceMap(cm::C, convert::Function) where {C <: Gen.ChoiceMap} = new{C}(cm, convert)
end
ConvertValueChoiceMap(::EmptyAddressTree, convert::Function) = EmptyAddressTree()
ConvertValueChoiceMap(v::Value, convert::Function) = Value(convert(get_value(v)))
Gen.get_subtree(c::ConvertValueChoiceMap, a) = ConvertValueChoiceMap(get_subtree(c.cm, a), c.convert)
Gen.get_subtrees_shallow(c::ConvertValueChoiceMap) = (
    (a, ConvertValueChoiceMap(sub, c.convert)) for (a, sub) in get_subtrees_shallow(c.cm)
)

"""
    values_to_abstract(world::World, cm::ChoiceMap)

Return a choicemap which converts all concrete OUPM object values in `cm` to abstract form.
Will error if a concrete OUPM object appears as a value which is not associated with
an abstract object in the world.
"""
function values_to_abstract(world::World, cm::ChoiceMap)
    convert(val) = convert_to_abstract(world, val)
    ConvertValueChoiceMap(cm, convert)
end

"""
    values_to_abstract!(world::World, cm::ChoiceMap)

Return a choicemap which converts all concrete OUPM object values in `cm` to abstract form.
Will create a new association in the world if a concrete OUPM object appears as a value
which is not currently associated with an abstract object in the world.
"""
function values_to_abstract!(world::World, cm::ChoiceMap)
    convert(val) = convert_to_abstract!(world, val)
    ConvertValueChoiceMap(cm, convert)
end

"""
    values_to_concrete(world::World, cm::ChoiceMap)

Return a choicemap which converts all abstract OUPM object values in `cm` to concrete form.
Will error if an abstract OUPM object appears as a value which is not associated with
an concrete object in the world.
"""
function values_to_concrete(world::World, cm::ChoiceMap)
    convert(val) = convert_to_concrete(world, val)
    ConvertValueChoiceMap(cm, convert)
end