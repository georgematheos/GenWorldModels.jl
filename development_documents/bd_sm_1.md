# Birth/Death & Split/Merge

### "Low level": Update Spec
At a fairly low level in the software, "birth", "death", "split", and "merge"
will be custom update specs for a `UsingWor$$ld` model.
These update specs will be constructed via some higher level interface (perhaps
a hook in the involution DSL).

#### Core functionality
In intuitive terms, we say these inference moves "insert or delete objects"
in the open universe model.  To put this in more formal terms, we will say:

1. An "object" is identified using an index in some indexing scheme.  I will first
write about the simple case in which the index is a simple number ``i``.
2. For each type of object, there are some number of "latent variables".
In `UsingWorld`, these are all memoized generative function calls with the object
as the argument.  We will say we have ``n`` latent variables for each
object, ``x_1(i), ..., x_n(i)``.
3. When we "insert" an object at index ``j``, this means that we increase the index of the latent
variables for all obects ``i \geq j``.  This means that after the index we have a new
collection of latent variables ``x_k'(i)`` such that for every latent variable ``x_k``,
``\forall i < j, x_k'(i) = x_k(i)`` and ``\forall i \geq j, x_k'(i + 1) = x_k(i)``.
Just saying we are "inserting" an object at index ``j`` does not specify what
the value of ``x_k'(j)`` is.  In practice, this will be regenerated or constrained somehow.
4. When we "delete" an object, we likewise decrease the index of all latent variables with higher
index, in the opposite move as the above.

It is clear how "birth" and "death" moves insert or delete objects.  For the current discussion,
I am viewing a "split" move as one that deletes 1 object and then inserts 2 new ones,
and a "merge" move as one that deletes 2 objects then inserts 1 new one.
The ability to somehow "combine the latent variables" when merging is not core
to my current definition of "merge"ing used in this section of the document.

#### Identifying latents in the model vs in the update
To automate these updates to some extent, we at some point need to
communicate to `UsingWorld` which memoized generative functions should be
viewed as latent variables for each type of object.

##### Identify latents in the update
One option
is to communicate this as part of the update spec.  For instance,
we could have the specs look like
```julia
Birth(;
    idx=4,
    mgfs=(:volume, :frequency, :length)
)
```
Here, we specify that we are going to birth an object at index `4` among
the class of objects which have 3 latent variables, identified with addresses
`:volume`, `:frequency`, and `:length`.

##### Identify latents in the model
However, identifying latents in the updates is needlessly verbose, especially
since in a model each memoized generative function will only take arguments
which are a certain "type" of object.  Having to specify what all the MGFs
are for a certain type in every update leads to a lot of redundant (and potentially
error-prone) code.

Another approach would be to declare in the construction of the `UsingWorld`
object that there are some number of types of objects, and which memoized generative
functions take each as an argument.  For example, we could construct
a `UsingWorld` instance via
```julia
UsingWorld(
    kernel,
    MGFSpec(:volume, generate_volume; argtype=:audiosource),
    MGFSpec(:frequency, generate_frequency; argtype=:audiosource),
    MGFSpec(:length, generate_length; argtype=:audiosource),
)
```
Then we can specify the birth move above via:
```julia
Birth(;idx=4, objtype=:audiosource)
```

I think this second approach of identifying the latent variables for each argtype
in the model rather than in each update is perferable, since it results
in neater code for update algorithms and less chance of the user making a mistake
by listing the wrong MGFs in a update specification.
We could alternatively achieve this via the construction of some objects
that understand object types at inference-time, rather than modeling-time,
but it seems to me that an understanding of what the object types are
must be present at modeling time, so I don't see any advantage from doing this.

### Origins
The type of birth and death reindexing I have described so far has the major
shortcoming that it assumes all objects are identified by a single integer
index.  There are arguably at least 2 shortcomings:
1. Sometimes it does not make sense to "index" the objects; we would rather simply have unique identifiers for each.
2. Sometimes the natural way to describe objects is with several
indices/identifiers, not one--and the standard one-index reindexing rule
for birth/death or split/merge result in different behavior than we want.

Shortcoming (1) is out of the scope of this project, and should be handled
in future work on "abstract partial instantiations" and Bayesian nonparametrics.
I don't think that shortcoming (1) vastly limits the modeling flexibility
of the system; while it makes it impossible to express some models
in the most natural way possible, it should still be possible to approximate
desired models with the system most of the time.

However, shortcoming (2) is more significant, and I think that if it is not dealt
with, it severely limits the utility of the birth/death and split/merge
primitives.  As an example of where this can arise, consider
an open universe model to put a prior over the space of PCFGs.
We might have our model look something along the lines of

1. There are some number of nonterminals.
2. For each nonterminal, there are some number of production rules.

To identify nonterminals, our indexing scheme seems sufficient:
we can have nonterminal `NT(1)` or `NT(10)`.  However,
it would be unnatural to identify production rules with a single
index.  What is the production rule `PR(104)`?  Which nonterminal
is this a production rule for?  It would be much more natural
to identify production rules by both specifying what nonterminal it is
a production rule for, and what index it is among production rules for that
nonterminal.  At least intuitively, it would be nicer to
identify the 2nd PR for the 8th NT via `PR(NT(8), 2)`, not
``PR(2 + \Sigma_{i=1}^7{(\text{Num PRs for NT(i)})})``.

While I think it is possible to implement the needed functionality using
single-indexing, my feeling is that at the very least that there should
be a user-facing interface where users can fairly easily use the
`PR(NT(8), 2)` syntax.  Also, users should be able to specify updates
along the lines of:
```
BIRTH a new NT at index 5;
MOVE PR(NT(4), 2) and PR(NT(1), 6)
  to be PRs for the new NT.
```
or
```
DEATH NT(5);
MOVE
  PR(NT(5), 1) --> PR(NT(4), 2)
  PR(NT(5), 2) --> PR(NT(1), 6)
```
or
```
SPLIT NT(5) into NT(4) and NT(8);
MOVE
  PR(NT(5), 1) --> PR(NT(4), 1)
  PR(NT(5), 2) --> PR(NT(4), 2)
  PR(NT(5), 3) --> PR(NT(8), 1)
  PR(NT(5), 4) --> PR(NT(4), 3)
```
or
```
MERGE NT(4) and NT(8)
  s.t. the PRs for NT(4) end up at indices
    (1, 2, 4)
  and the PRs for NT(8) end up at indices
    (3,)
```

These updates all consist of a 2 steps:
1. Insert and/or delete some of the parent object type (NTs)
2. Move/insert/delete children (PRs)

We could break this down further as:
1. Insert and/or delete some of the parent object type (NTs)
2. Insert and delete some children variables (PRs), keeping
   the latents for the deleted children "off to the side" and not
   yet generating latents for the inserted children.
3. Fill in the latents for the inserted children with those for the deleted
   children, with some specific alignment between the inserted and deleted objects.

### Possible issue in implementation: updating subtraces with new args
Our goal is to effectively move the subtraces for latent variables for objects
as efficiently as possible.  However, this involves changing the argument
of the object, and thus for correctness,
when using a black box memoized generative function, we have to call
`update` on every moved subtrace to update the argument.

If the memoized generative function uses the argument index somehow, there
is no way to avoid this.  However, if it does not, this is a needless update call.

If the argument is not used:
- If the MGF is in the dynamic DSL, the update will result in fully revisiting
all calls in the subtrace, which is very slow.
- If the MGF is in the static DSL, ideally, static compilation would realize
that the changed argument will not result in any changes, and turn each update
into an instant return of `(old_tr, 0., NoChange(), EmptyChoiceMap())`.  However,
since the `world` object may have an argdiff if other things are being updated,
the static DSL may not realize that it can immediately return this!
For this to happen, we'd have to have enough information in the
world argdiff type for the static DSL to realize none of the values the current
trace looks up have changed.  This may be possible, but it seems to me
that getting this to work would fairly complicated.

One of the advantages of `UsingWorld` is that even when using dynamic DSL
functions as MGFs, we can get quite a bit of incremental computation out
of the `lookup_or_generate` boundaries between generative functions.
In other words, `UsingWorld` can typically manage to only update
the minimum possible number of dynamic DSL calls in the world needed
for an update the user specifies (ie. we can calculate markov blankets
in the graph where each node is an entire memoized generative function call).
It would be nice to extend this advantage to this situation,
and not have to update all the dynamic DSL
calls for latent variables for objects that are being split/merged.

#### A possible solution
One way to achieve this would be to make sure that any time a MGF looks at its
arguments, it does so via a `lookup_or_generate` call.  Then,
we could detect which generative functions need to be updated due to a change
in their arguments using the usual `update` algorithm!

One way to do this would be to pass in arguments to MGFs in some sort
of "encoded" form, so that the only way to "decode" is to call
```
decoded ~ lookup_or_generate(world[:decode][encoded_argument])
```

### The "identifier trick"

#### As a way of "encoding" arguments

One way we could accomplish this type of encoding would be that every time
we call a MGF on an argument for the first time, we
1. Generate a UUID for the argument
2. Store the entry `UUID <--> argument` in a bidirectional
lookup table for the world object
3. Pass in the UUID to the MGF

Every subsequent time we call an MGF on that argument, we lookup the associated
UUID and pass that in as an argument instead.

When the user calls `lookup_or_generate(world[:decode][UUID])`, the world
will look up the argument in the lookup table.  Thus the only way for a MGF
to actually access its' argument's value is to look it up in the world, as we need
for incremental computation to occur in our update algorithm!

#### As a way to make it easy to move latent variables
Using this "identifier trick" also makes it easy to implement
the birth/death and split/merge moves.  Whenever we want
to "move a latent variable" to a new argument index, we simply
change the lookup table's association of index arguments and their
corresponding UUIDs.  Any MGF which uses the actual index argument
will be updated because it has a call to `lookup_or_generate(world[:decode][UUID])`.
But any MGF which does not need the actual index will be able to avoid
updating the subtraces which are being moved.

Any calls to `lookup_or_generate(world[:latent_var_mgf][index_arg])` will
be updated when the MGF is moved since the world argdiff will communicate
that the value for `:latent_var_mgf` called on indexed arguments has changed
for all the indices which have had their identifier associations changed.

#### That said...users need to know where they use indices!

While this system of substituting in UUIDs will make it possible to build
a system which can automatically implement much of the bookkeeping needed
for split/merge and birth/death updates, there are some things it will not
be able to do automatically.  In particular, the only place this system
can automatically "move subtraces" is at the boundaries of MGF calls which
take an objeect as the argument.  When using the identifier trick,
we get some extra milage out of this, because MGFs called with a UUID
can pass this UUID around and references to this UUID be changed when
we change data associations.  However, anywhere in the code when we
refer to objects using indices, users will need to update
manually.  For example, if an object is sampled by drawing
an index `idx ~ uniform_discrete(1, num_objects)`, if this index needs
to change for a birth/death or split/merge move (eg. because we added/removed
an object at a lower index than the previously sampled `idx`), 
the user will have to manually change this index.

We could implement special case behavior to help alleviate some of this, eg.
a special distribution which samples an object uniformly at random
from the world directly in the UUID
representation. But ultimately, so long as we retain the notion of "indices"
that the user can sample using standard Gen distributions, there will be
the potential for users to refer to objects using these indices in black-box
models which our system cannot automatically update.

### Validity of the "identifier trick"
How can we be sure that this "identifier trick" is valid?  The key is that
in any `UsingWorld` trace, there is a concrete value associated with every identifer.
If we substituted every appearance of the identifier with this concrete value, we
would get a trace with no mention of identifiers.  The distribution represented by
the `UsingWorld` model is the distribution over that de-identifiered trace.
If we want to be very careful, we could override the `==` function for traces
so that the comparison occurs between the de-identifiered versions.

The only things a user can do with the identifer objects are:
1. Make MGF calls with the object as argument
2. Compare the equality of identifier objects
3. Lookup the index associated with the identifier in the world
4. Pass around the identifier object so other parts of the program can
  perform these operations

If 1-3 are valid in all places in a `UsingWorld` model, certainly (4) is valid,
since it just allows us to do 1-3 at other parts of the `UsingWorld` model.

(3) is valid because this is treated as a regular `lookup_or_generate` call in the world,
and thus we can correctly update the index looked up for each identifier whenever associations change.

Since there is always a 1-1 association between identifiers and indexed objects,
(1) is a valid operation, since it is semantically equivalent to looking up the index of an object
using (3) and then calling the MGF on that index.

(2) is always valid because of this 1-1 association, so long as we are comparing objects within the context
of a `UsingWorld` model.

Note that outside the context of a single `UsingWorld` model, we have to be somewhat more careful.
For instance, we can't so easily compare 2 identifier objects from different traces, since they might have
different associations.  Perhaps we should disallow the `==` function and only allow calls to
`equal_in_context(id1, id2, world)`.
If an identifier object is returned by a `UsingWorld`
model and used in a parent generative function, I don't see any harm that could arise,
under the assumption that every identifier is truly unique.  The identifier is pretty much
meaningless outside the `UsingWorld` model, and because of this, I don't see how invalid
semantics could arise.