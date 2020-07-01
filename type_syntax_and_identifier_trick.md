# Gen OUPM `@type` mechanism

Here is how I propose to implement the "identifier trick" to allow for more
efficient updates in open-universe inference.  

### Attempt 1: "Move subtrace" solution
Say we have a `UsingWorld` model which conceptually has some type of object
we are running birth/death or split/merge inference over.
These objects may be the lookup keys for memoized generative functins in the world.
As we run inference, we want to insert and delete objects at different indices, which means that many objects are "having
their indices changed" to make room for the new objects or "fill in the gaps" left by deleted objects.
To accomplish this efficiently, it makes sense to "move" the currently
generated subtraces for the memoized generative functions that have an object whose index has changed as
an argument.  By "moving", I mean we can just change the key for this subtrace without generating a new one.

#### The shortcoming of this
The major issue with this is: what happens if there are tons of memoized generative
functions which take this object as an argument?  If we have to move the subtraces
for each one, this might be a very expensive operation.
It is `O(N * M)`, where `N` is the number
of objects which must be reindexed, and `M` is the number of memoized generative functions
taking each object as an argument.

Here, we introduce a solution called "the identifier trick" which reduces this overhead to `O(N)`.
(Future work may implement "abstract partial instantiations" or something analagous
to make some of these updates `O(1)`.)

### A syntax for declaring types
To help automate the reindexing process, we will allow users to declare
different "open universe types", using a `@type` macro.  For example:
```julia
@type AudioSource
```
These objects can be thought of as simply wrapping an integer index:
```julia
AudioSource(4) # the 4th AudioSource
```

When we have a memoized generative function which only takes objects
of a certain type as argument, we can annotate this:
```julia
@gen function sound_from_source(world, source::AudioSource)
    # ...
end
```

This makes it possible to add a syntax along the lines of:
```julia
new_tr = birth_update(old_tr; birth_type=AudioSource, birth_idx=2, num_statement_address=:kernel => :num_sources)
```
to specify an update that adds an audio source at index 2, modifying the "number statement" at address `:kernel => :num_sources`.
This `birth_update` function can automatically reindex the subtraces for every memoized generative function
in the world which takes an argument annotated as being an `AudioSource`.
(In practice, we will use a different syntax for specifying an MH kernel which proposes a birth move; this
is an oversimplified example.)

### The identifier trick

To avoid needing to move the subtraces for memoized generative function,
I propose that we actually have the lookup keys for the memoized generative functions
not be an `AudioSource(index)`, but instead be an `AudioSource(identifier)`.
When we call `lookup_or_generate(AudioSource(index))`, this automatically
gets converted into a call `lookup_or_generate(AudioSource(identifier))`.

We add a 2-way lookup table to the `world` state, mapping from indices to identifiers.
Using this, we get a function `get_id(AudioSource(index))` and `get_idx(AudioSource(identifier))`.

Since all the memoized generative function keys are for the `identifier` representation,
to change the index of an `AudioSource` object, we simply need to update our lookup table.
For instance, if we delete the second audio source, we'd need to move `AudioSource(3)` to become `AudioSource(2)`,
and currently `get_id(AudioSource(3)) = #AAK21`, we simply update the lookup table so that
`get_idx(AudioSource(#AAK21)) = 2` and `get_id(AudioSource(2)) = #AAK21`.  Now,
all the memoized generative functions which have `AudioSource(#AAK21)` as their key
are calls for `AudioSource(2)` rather than `AudioSource(3)`.  Viola!  In one change, we moved subtraces
for arbitrarily many memoized generative functions!

#### Validity

The key to this is that from the user's perspective, there is no such thing as an "identifier".
They just use `AudioSources` with indices, and happen to observe surprisingly fast reindexing updates.
Therefore there's next to no need to argue for the validity of this: we have not changed the semantics
of the model at all.

#### Where in the program do identifiers appear?
Since we transform the arguments to memoized generative functions to the identifier represenation,
these MGF call traces will have the identifier representation of the object, not the indexed
representation.  I don't think this is an issue, but it informs the type of interface we need
to design--for example, users can only get the index of an object if they provide the world.

##### Question: should choicemaps use identifiers or indices?
We could transform the keys to the MGFs to appear in the index representation in the `UsingWorld` choicemap.
I think this is probably the right call, but I also wonder if it's important to make it apparent
in the exposed representation that these are in fact stored in the identifier representation.

### Implementation Details - Macroexpansion for `@type` statements
I'm not totally settled on what this will look like, but one way we could expand
the `@type` macro to support using either the identifier or indexing representation
would be to have
```julia
@type AudioSource
```
macroexpand into something along the lines of
```julia
abstract type AudioSource <: OUPMType end
struct _ID_AudioSource <: AudioSource
    id::Identifier
end
struct _IDX_AudioSource <: AudioSource
    idx::Int
end
AudioSource(idx::Int) = _IDX_AudioSource(idx)

get_index(as::_IDX_AudioSource, ::World) = as.idx
get_index(as::_ID_AudioSource, w::World) = get_idx_representation(as, w).idx

# Error messages make it clear we need the world for context to perform some operations, in case we're using identifiers.
get_index(as::AudioSource) = error("Must pass in world for context in case this is in the identifier representation!.")
Base.:(==)(::_ID_AudioSource, ::IDX_AudioSource) = error("Cannot compare ID and IDX representations without context.  Try `is_equal_in_context(a1, a2, world))")
Base.:(==)(a1::IDX_AudioSource, a2::_ID_AudioSource) = ==(a2, a1)
```

### Using Identifier Sampling for Further Performance Improvements
Even if we use identifiers in all the lookup keys for subtraces in the world,
there may still be references to indices within the bodies of generative functions
in the model which users will need to explicitly specify to change the indices of
during reindexing.  For example, if we sample an audio source, we would write
```julia
num_sources ~ lookup_or_generate(world[:num_audio_sources][()])
source_idx ~ uniform_discrete(1, num_sources)
source = AudioSource(source_idx)
```
If we need to reindex the sampled `source`, it means we will have to update `source_idx`.
If we want to, we can introduce a special distribution for sampling
an object in its identifier form.  For example, this could look something like
```julia
num_sources ~ lookup_or_generate(world[:num_audio_sources][()])
source ~ SampleInIdentifierRep(AudioSource, uniform_discrete)(world, 1, num_sources)
@assert source isa _ID_AudioSource
```
Here, we add a "distribution combinator" `SampleInIdentifierRep` which is constructed with the type
of object to sample, and the distribution for sampling the index of the object.  It is then called
with the `world`, and the arguments for the underlying distribution.  Internally, it will sample from this index
representation (ie. `idx ~ uniform_discrete(1, num_sources)`), then construct an object with this index,
and then get the identifier for it in the world, and return this identifier representation.

If we do this, then update the trace to change the correspondance between identifiers and indices,
we will retain the same sampled identifier, rather than the same sampled index.  This means we don't have
to update this call during reindexing.

Users who use this will have to understand to some extent how the identifier system works, so they understand
how their updates will operate.  So this would be an "advanced feature" if we implemented it.

#### This probably isn't too important, though
While I wanted to mention we could support this, I'm
not convinced this is too important, since most of the time when we are reindexing, we have to change
the number statement (ie. `num_sources`).  Thus, at least in the sort of use case in this example,
this isn't an asymptotic performance improvement, since we still need to recalculate the probability
for the sample.

### OUPM types with origins
I think that we will want to support some notion of an "origin statement" as in BLOG.
I am still hashing out the details of how this looks, but an example where we want this is
in designing PCFGs.  We want a type `Nonterminal`, and for each nonterminal, we want some number
of production rules.  We might write this:
```julia
@type Nonterminal
@type ProductionRule(::Nonterminal)
```
We then have a seperate number statement for production rules for each nonterminal,
and this way when we split/merge or birth/death a production rule for one nonterminal,
we only have to reinded the production rules for that nonterminal, and not the others.

I will think more about this and follow up about it.