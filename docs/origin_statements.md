# Supporting OUPM Types with "Origins"

### Origin vs Latent Variable

We use "origins" to describe a relationship between an object of an OUPM type
and some tuple of "origin objects".  Conceptually, this relationship is that the existence
of the "child object" depends upon the existence of the "origin objects".
In a generative sense, we think of first generating all the origin objects--then for
each tuple of these origin objects, generating some number of children.

Conversely, we can also represent some relationships between an object and others
via sampling an object as a latent object.  Eg. we could have something like:
```julia
@gen parent(world, obj::ObjType1) = ObjType2({:idx} ~ uniform_discrete(1, @lw num_objs[()]))
```
where `@lw` inserts a traced lookup from the world.

This is similar to the origin statement, except the generative process is reversed: we first
generate the children, then select a parent for each.  Because origin statements
imply that the existence of the children is dependent upon the parents, the meaning of open universe
inference moves like "split" and "merge" are also different when they are used.

### Referring to an object with an origin

Since objects with an origin "come from" that origin, we write out the origin objects
as part of the "name" of a child object.  For example, in the aircraft/timestep/blips example,
we might write
```julia
Blip((Aircraft(2), Timestep(5)), 1)
```
to denote "the first blip coming from Aircraft 2 at timestep 5".

## OUPM Moves

For simplicity, we will assume there are 2 types of object, `A` and `B`,
where each `B` has an `A` as origin.

### Birth/Death

#### Birth `A`
If we birth an `A`, by default it has no `B` which it is a parent for.

#### Death `A`
We can only delete an `A` which has no "instantiated" children, ie. no children which we have
generated identifiers for.  If we want to delete an `A` with children,
we first have to move all the instantiated children to have a different object as origin.

#### Birth `B`
When we birth a `B`, we have to specify what `A` it will have as a parent.
Then all the `B`s with this `A` are reindexed, but `B`s with other `A`s as origin
don't need to be reindexed.

#### Death `B`
This is similar to the birth.

### Split/Merge

#### Split `A`
For every `B` parametrized by the `A` being split, the split move must specify
which of the new `A` objects the `B` is "moved to" having as its origin.
The user must specify which indices for each of these objects each item ends up at.

#### Merge `A`
All `B` objects with either `A` object being merged as its origin will be changed
to have the new `A` object as its origin.  The move must specify what indices each of these
items ends up at.

#### Split `B` or Merge `B`
Similar to the B birth and death in that the move is the same as
in the no-origin case, except we only reindex the items with the same origin.
Note that we can only merge 2 objects with the same origin, and when we split
an object each new object gets the origin of the original one.

### Move

#### Move `A`
All the `B` objects should remain parametrized by the `A` object with the same "identity",
even though the index of the `A` may be changed now.

#### Move `B`
We can either move `B` by just changing its index, or by changing both its index and its origin.
In this second case, we have to reindex both the `B`s for the old and for the new origin.

### Automatic "dumb" moves
We could potentially have `UsingWorld` automatically make some of the decisions during OUPM move updates.
For instance, when splitting an `A`, it could automatically randomly divide the children between the two
new objects.  Or we could "move" an object to a new origin, and have its new index be randomly selected
automatically.

However, to implement this, I think I will need to improve the "externally constrained address" system.
We will probably also need to maintain a map from each object to all its "children".  I am going to push this
off to a "second pass".

## Software Representation

There are a few possible ways an object can be represented.

#### Concrete representation

This is the form in which all origin objects are in their concrete representation,
and we store the INDEX of the object for this origin.  Eg.
```julia
Blip((Aircraft(2), Timestep(5)), 1)
```

#### Fully abstract representation

This is the form in which we only store an identifier for the object.  Eg.
```julia
Blip("##Blip#276")
```

#### Origin-abstract representation

This is a form where we store the origin, where each origin is in its fully abstract representation,
and we store the concrete index of the object for this origin.  Eg.
```julia
Blip(
    (Aircraft("##Aircraft#277"), Timestep("##Timestep#278")),
    1
)
```

#### Mixed representation

We could also store an origin where some items are mixed or fully abstract, and others are concrete,
and store the index concretely.  Eg.
```julia
Blip(
    (Aircraft("##Aircraft#277"), Timestep(4)),
    1
)
```

### In UsingWorld
The ID table will store the mapping between objects in fully abstract form
and in origin-abstract form.

When calling a MGF with an object, the object is converted into fully abstract form
automatically.

### Accessing the index and origin.

These can be accessed via lookups:
```julia
index ~ lookup_or_generate(world[:index][Blip("##Blip#276")])
origin ~ lookup_or_generate(world[:origin][Blip("##Blip#276")])
```

### Getting the fully abstract form
```julia
abstract_object ~ lookup_or_generate(world[:abstract][Blip((Aircraft(2), Timestep(3)), 1)])
```

This will internally get the abstract form for each origin object via a recursive call to `@w abstract[...]`.

## Julia type for objects
I think that for simplicity I should store objects as a single Julia type, parametrized by the object type.
This will look something like
```julia
abstract type OUPMObject{T} end
struct AbstractOUPMObject{T} <: OUPMObject{T}
    id::Symbol
end
struct ConcreteIndexOUPMObject{T} <: OUPMObject{T}
    origin::Tuple
    idx::Int
end
```

When we call `@type`, this will do something along the lines of:
```julia
@type Blip
# expands into
Blip = OUPMObject{:Blip}
Blip(origin::Tuple, idx::Int) = ConcreteIndexOUPMObject{:Blip}(origin, idx)
```

It is possible that by annotating objects with the types of the origins may allow for
improved performance, but I'm not going to worry about that for now.

## Where does abstract <--> concrete conversion occur?

There are 2 types of conversion: conversion within the model, and conversion
for communication between the model and the "outside code".

#### Within the model

All conversion within the model passes through one of the following `lookup_or_generate` calls:
- `lookup_or_generate(world[:index][abstract_object])`
- `lookup_or_generate(world[:origin][abstract_object])`
- `lookup_or_generate(world[:concrete][abstract_object])`
- `lookup_or_generate(world[:abstract][concrete_index_object])`

`lookup_or_generate` will automatically insert calls to `lookup_or_generate(world[:abstract][...])` in the following situations:
- For any call to `lookup_or_generate(world[...][concrete_index_object])`, the concrete indexed object is converted to its abstract form.
- For any call to `lookup_or_generate(world[:abstract][concrete_index_object])` where the given `concrete_index_object` has objects in its origin
  which are in concrete form, all these concrete origin objects will be converted to abstract.

#### Between the model and the rest of the code

These cannot create new abstract objects:
- `discard`
- `externally_constrained_addresses`
- `choicemap`

This conversion may require creating new abstract objects:
- `update_spec`

If abstract objects are created during the conversion for the `update_spec`, since there is an update
provided for it, there will be an error at the end if we didn't need to generate this address.
So it should be impossible to generate an abstract object in this conversion which ends up being unused.

### Lookup count tracking

To integrate nicely with the lookup count and dependency graph tracking, we need to
- add a table for lookups of each instantiated object when they become instantiated
- remove this table when an object becomes uninstantiated

#### adding tables

The only place we can generate an abstract object is via a call to `generate_abstract_object!`.
This function will handle creating a call for all the concrete -> abstract and abstract -> concrete lookups possible.

It is possible to instantiate a new concrete object during a OUPM Move which associates a previously generated
abstract object with a new concrete object.  Whenever this occurs, we need to create calls for the
corresponding concrete -> abstract call in the lookup table.

#### removing tables

One place we may remove tables is if during our end-of-update count adjustment, we discover that
the count for looking up a certain concrete --> abstract call has dropped to zero.  In this case,
we should remove all calls for both the concrete --> abstract and the abstract --> concrete.  Further,
we only remove these tables when we remove the last concrete --> abstract call.  (So long as the abstract objects
exist in the model, it is possible that an update could lead to an abstract --> concrete call being looked up.
Since the `has_val` for these abstract --> concrete calls never became false, these new calls would not lead to a new
table addition.  So we can only remove the abstract --> concrete tables when we know there are no more abstract objects
present.)  I think it is fine to do this as part of the normal `note_lookup_removed` call.

Another place we may need to remove tables is when performing an OUPM move.  When we do this,
some concrete objects and some abstracts may become "uninstantiated" in that they no longer correspond
to any object of the other type.  We should remove the lookups for these at this time.

## TODOs
- `lookup_or_generate` conversion
- choicemap/discard/externally_constrained_addrs/update_spec conversion
- proper lookup table tracking
- something with diffs, probably
- OUPM moves