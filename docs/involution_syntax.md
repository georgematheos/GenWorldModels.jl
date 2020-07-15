# Birth/Death and Split/Merge in the Involution DSL

To specify birth/death and split/merge, users will write a proposal
that determines the indices specifying the move.  They then write an
involution function which takes the choicemap from this proposal
and implements the split/merge or birth/death move it specifies.

### Do we need a seperate involution?
In many situations, we could probably get by via some sort of "default involution"
which looks for some specific addresses in the proposal choicemap, and understands
these as specifying the update.  For example,
we could have the default address `:do_birth` specify
whether we do a birth or death move in a birth/death proposal,
and have addresses `:birth_idx` and `:death_idx` specify the index
of the object to create or delete.  However, this doesn't cover all possible cases.
For instance, what if our decision of the `birth_idx` is ultimately not drawn
from a distribution, but is instead a function of several other distribution draws?
To handle cases like this, I want to offer as much flexibility as is possible,
which leads me to want to allow users to write involutions implementing these moves.

However, it may make sense to provide default implementations for common cases.

## Semantics for reverse constrained addresses
When we specify a "birth" move for object type `T` at index `idx`, this means:
1. Move all the MGF calls on `T(j)` to be the call for `T(j+1)` for every `j >= idx`.

Note that this means there is no longer a subtrace for any MGF call on `T(idx)`.
If the updated trace has a nonzero number of MGF lookups for `T(idx)`, this means that
the calls for `T(idx)` will be `generate`d from scratch (possibly with any constraints
specified in the update).

This raises some ambiguity in the reversing "death" move, which deletes this subtrace.
For the "death" move to correctly calculate the weight, it needs to know which addresses
are constrained during generation.  (The unconstrained addresses should have the "regenerate"
weight, whereas the constrained addresses should have the "update" weight.)

In most situations we can deal with this because on any pass through the
involution function written in the involution DSL, we will encounter a
`@read_*_from_model(...)` call for every address we constrain (via `@write_*_to_model`)
in in the reverse move.
The reason for this is that the "backward proposal" the involution produces will need
to fully encode the value for these addresses.  If we constrain a value in the reverse move,
the value we put there will have to be in the proposal--so we need to look up the current
value from the current model to know what value has to be in the proposal for the reverse move.
This argument holds if the only way for us to know what the reverse move should propose
is by looking up this address in the current trace.
This holds if and only if we don't constrain any variables which are deterministic
given all other variables.  If we do constrain 
deterministic variables, we might not need to `@read_from_model` every variable
we constrain, since we can be certain what the value of the variable is by looking at the other
variables it is a deterministic function of.

Another issue is that we could `@read_from_model` some addresses we don't actually use for anything.
In this case, the model might mistakenly think we constrain something we don't!

To avoid these issues, I am going to propose a new involution DSL for use with OUPM inference that
requires users write models satisfying the fact that all values which are constrained in the model
are `@read_from_model` in the reverse direction--and these are the only addresses constrained.
One simple way to guarantee this is that it will always hold if

1. the model does not have any deterministic variables

and

2. the only calls to `@read_from_model` are for addresses which must get written to the proposal for the involution to be valid

### Checking for safety!
If we have `check=true` when we call the involution mh variant,
as part of the reverse-involution correctness check, we can confirm that the addresses
which are constrained during the reverse move are exactly those which were `@read_from_model` in the forward move,
and throw an error if this is not satisfied.

## Syntax
We add the following macros to the involution DSL:
```julia
@birth(ObjectType, idx)
```
adds an object at the given index of the given type.
The reverse move is
```julia
@death(ObjectType, idx)
```

For split/merge, we have
```julia
@split(ObjectType, from_idx, to_idx1, to_idx2)
```
and
```julia
@merge(ObjectType, to_idx, from_idx1, from_idx2)
```

By default, any new objects are entirely unconstrained.  However, users may also specify constraints
for the new objects within the involution. If MGF calls on these
objects are `lookup_or_generate`d in the new trace, then the calls will be `generate`d using whatever
constraints the user specifies in the involution, with all unconstrained addresses being generated
according to the internal proposal distribution, as usual.  (For details on how the reverse
weight is calculated, see the above.)

In addition to `@split/@merge` and `@birth/@death`, we provide the move
```julia
@move(ObjectType, from_idx, to_idx)
```
This is effectively the same as calling
```julia
@death(ObjectType, from_idx)
@birth(ObjectType, to_idx)
```
and then fully constraining the object at `to_idx` to be identical to that
which had previously been at `from_idx`.

#### Multiple moves

Multiple moves can be specified at once.  These moves are applied in the order
they are encountered, gradually updating the indices as they go.  As an example, to do 
a 3-way swap of object indices, so `(1 --> 2, 2 --> 3, 3 --> 1)`, we could use
```julia
@move(T, 1, 2) # 1' = 2, 2' = 1, 3' = 3
@move(T, 1, 3) # 1'' = 3' = 3, 2'' = 2' = 1, 3'' = 1' = 2
# so overall, 1'' = 3, 2'' = 1, 3'' = 2, as desired
```

## Parsing the macros
To implement this, I will make a fork of the involution DSL code for use
on a `UsingWorld` model.  I will change the following things
1. Instead of the state accumulating to a `constraints::ChoiceMap`, the state will accumulate a `spec::UpdateSpec`.
   1. When we encounter a `@write_to_model` call, we will simply push the new choice to the `spec` the way we would for the `constraints`.
   2. The first time we encounter one of the special OUPM macros, we will change our `spec` object from a `DynamicChoiceMap`
   to a `OUPMUpdateSpec`.  This is a `CustomUpdateSpec` which has a list of "moves" to do for the objects in an open universe model,
   and has a sub `UpdateSpec` containing the update specs to update the kernel and the MGF calls.  We will store the old dynamic choicemap
   as the sub-updatespec, and add the special macro we encountered to the queue of special open universe updates.
   3. Every subsequent OUPM macro we encounter will be added to the queue of open universe updates in the `OUPMUpdateSpec`.
2. We will also accumulate a `reverse_constrained_selection::Selection`.
   1. Every time we encounter a `@read_from_model` statement, we will add the read address to this selection.

The code for Jacobian determinant calculations will not be altered.  I don't think it needs to be, since
all the changes to the choicemap of the model induced by the open universe moves are just changes of the argument to
the memoized generative functions.  Since these arguments are discrete, this cannot effect the Jacobian.
Any "regenerated" values will yield a row in the Jacobian matrix with a 1 at one spot and 0s at all others
(since the proposed value is exactly the resulting value), and thus will not effect the Jacobian determinant.
Any constrained values will require a `@write_to_model` call and will thus be dealt with in the usual way.
Any concerns arising from the fact that it is possible an update could result in branches of the program disappearing
are alleviated because the reverse move will have to constrain these branches, and thus insert `@write_to_proposal`
calls that will ensure valid Jacobian calculations occur.