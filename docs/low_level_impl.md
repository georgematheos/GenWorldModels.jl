# Low level implementation for open universe types and updates

### Tell the model the open-universe types & the latents

#### Declare OUPM types

At the beginning of the model, we declare the open universe types as Julia types.
Later we can add syntactic sugar, but the underlying representation is that
each such object stores either an identifier or an index:
```julia
struct AudioSource{T} <: OUPMType{T}
  id::T
end
```

#### Restrict argtype of the latents

When we declare the memoized generative functions which are latent variables
for each OUPM type, we can provide a Julia type annotation for the argument
in the function signature:

```julia
@gen (static) function volume(world::World, source::AudioSource)
```

This is not strictly necessary at this point, but it may prevent errors
from being thrown.  In the future, we could use this to automatically detect
what MGFs are latents for what OUPM type, though right now we
will specify this explicitly when constructing the `UsingWorld` object.

#### Specify the argtypes of latents when constructing `UsingWorld`

We now construct `UsingWorld` using syntax like
```julia
UsingWorld(
    kernel,
    address => mem_gen_fn,
    address => mem_gen_fn,
    ...;
    [
    world_args=(...),
    oupm_types=(AudioSource,)
    ]
)
```
We add an optional keyword argument `oupm_types` telling the `UsingWorld`
instance what types we need to track identifiers for.  All these types
must be subtypes of `OUPMType`.

### `world` stores bidirectional identifier lookup table
When a `world` object is constructed from a `UsingWorld` instance
with associated open universe types, it will create a bidirectional
functional collection lookup table between identifiers and indices
for each associated OUPM type.

### Changing index --> identifier

In a generative function, we can call
```julia
source::AudioSource{UUID} ~ lookup_or_generate(world[AudioSource][idx])
```
to get the identifier representation of the audio source with index `idx`.
To support this, for every OUPM type `T` associated with the world, we will
add the namespace `Symbol(T)` to the supported calls.  This call
either looks up the identifier for the given `idx` if one exists, or
generates one and stores it in the world lookup table if none exists yet.
Note that this means each open universe type `T` must have a different name.

#### Automatic change at latent call

Whenever a call
```julia
val ~ lookup_or_generate(world[:mgf_address][AudioSource(idx::Int)])
```
is made, internally this is converted into code which has the same effect as
```julia
source ~ lookup_or_generate(world[AudioSource][idx])
val ~ lookup_or_generate(world[:mgf_address][source])
```

#### Automatic change at latent call for updates
Likewise, if an update specifies an update for address `:world => :mgf_address => AudioSource{Int}(idx)`,
this will internally be understood as an update for `:world => :mgf_address => ~lookup_or_generate(world[AudioSource][idx])`.
Note that the identifier for this index will be the identifier AFTER the OUPM updates have been applied to the world.
This ensures that any constraints specified for a given index are actually reflected in the choicemap
for that index after the update is done.

### Get the index associated with an OUPM object
In a generative function, we can get the index associated with
an object like `source::AudioSource{UUID}` by calling
```julia
idx ~ lookup_or_generate(world[:index][source])
```

To support this internally, we will augment world to
treat `:index` as a special namespace which has its value returned by triggering
a lookup in the lookup table.  It is an error if the identifier for
`source` is not recognized.

### OUPM updates

For each OUPM update, we will have an internal function like
```julia
_perform_birth(world, AudioSource, index)
```
This function will reindex the identifers in the update table as needed for the update.
It will also note in the `world.state.diffs` that each `Call(:index, AudioSource(id))`
has a diff.  (This ensures that index lookups get updated.)
We will also record in a new part of the update state which indices have had the associated
identifiers changed.  The function `OUPMType` will recognize this diff to know
when the identifier object it returns for a given index has changed.
The `OUPMType` function will also recognize when the index passed in has changed
in a way consistent with the world object's change, so it should still return a `NoChange`.

### Number statements
Currently users will have to manually update number statements
to be consistent with the OUPM move; we can automate this in the future.

### OUPM Update Spec
We implement the custom update spec
```julia
struct UpdateWithOUPMMoves <: CustomUpdateSpec
    moves::Tuple{Vararg{<:OUPMMove}}
    subspec::UpdateSpec
end
```
where we have
```julia
abstract type OUPMMove end
struct BirthMove <: OUPMMove
    type::Type{<:OUPMType}
    idx::Int
end
# ... other move types ...
```

### `UsingWorld` ChoiceMap
The address for any MGF call which has an `OUPMType{UUID}` as an argument
will be replaced by the corresponding `OUPMType{Int}` in the `UsingWorld` choicemap.

### "Reverse update spec"
When we return a "discard"/"reverse update spec", the namespace
will also substitute in the indexed objects for MGF calls (rather than
identifier objects).

If open universe moves are performed, the reverse update spec
will be an `UpdateWithOUPMMoves` where the "discard" is the subspec,
and the `moves` implement the reverse of the OUPM moves passed in.
In this case, the indices put for the MGF calls in the `subspec` will
be the indices for the state of the world BEFORE the forward moves were performed.
This ensures that when the reverse `UpdateWithOUPMMoves` is performed,
since the `subspec` is applied after the reversing `moves`, that the
`subspec` truly is a reverse-direction update spec.

### Places id/idx conversion can occur
1. When there is a call to `lookup_or_generate(world[AudioSource][idx])`
2. When there is a call to `lookup_or_generate(world[:index][AudioSource{UUID}(id)])`
3. When there is a call to `world[:address][AudioSource{Int}(idx)]`.  This gets converted to `world[:address][AudioSource{UUID}(id)]`.
4. When returning the `choicemap`, MGF call addresses are moved from `:address => AudioSource{UUID}(id)` to `:address => AudioSource{Int}(idx)`
5. When returning the `discard`, MGF call addresses are moved to the index for the state of the `id_table` before the update occurred
6. When an update spec for `UsingWorld` has something at `:address => AudioSource{Int}(idx)`, this is viewed as `:address => AudioSource{UUID}(id)`
7. When a `externally_constrained_addrs` selection for `UsingWorld` has something at `:address => AudioSource{Int}(idx)`, this is converted.

It should be an error if in an update to `UsingWorld`, the `externally_constrained_addrs` selection or the `update_spec` both specify
an update for an object in ID form and the same object in IDX form.

### Standardized IDX vs ID formats for updates

To retain valid Gen semantics, during an `update` with an update spec
of type `AddressTreeLeaf{<:Union{Value, SelectionLeaf})`, all the addresses
provided as MGF keys should be in the index representation--since this is the representation
exposed in the `UsingWorld` choicemap.  Likewise, the discard should be in index representation.

However, when we have a `UpdateWithOpenUniverseMoves`, as a custom update spec, we may
have whatever semantics we like for the underlying `subspec`.  Since I forsee that these custom
update spec objects will typically be constructed programatically (probably using some
sort of involution DSL), would be reasonable to have the `subspec` accept MGF keys in its addresses
in identifier form.  Since when we perform open universe moves, the indices change meaning, it
becomes slightly thorny to use the index form for specifying what moves should occur.
As a result, while I intend to allow users to use indices too in whatever interface is ultimately
exposed to them for this, I have in mind that I also want to support directly using the identifier form.

However, the cases where this is important are probably fairly rare, since usually when we are
doing an open universe move, if we're constraining anything, it's one of the newly created objects.
(It seems more unusual for us to constrain another object in the same move as we eg. do a split/merge.)
While I ultimately want to support this, for now I'm going to simply require that
all the addresses are in index form.

#### Implementation: `world` sees everything in identifier form
The `UsingWorld` combinator will convert the from the indexed form
to identifier form before passing the update spec and externally constrained
addresses into the `world` to update the calls there.  To do this, I will
need a few stages:
1. `begin_update!` will update the world args and perform open universe moves
   and return a new `world` object
2. The kernel will convert the spec and ext_const_addrs to identifier form.
   It might do this lazily by just wrapping the choicemaps in some other form.
3. `run_subtrace_updates!` will use this converted spec, etc., to update the world
4. then we run the kernel updates and finish up as usual


## TODOs
- Argdiff propagation for conversion in `world[:address][IDX_FORM]`
- Argdiff propagation for `world[AudioSource][idx]`
- Reverse update specs
- Thorough testing

Completed, but may be worth testing more carefully:
- Automatic conversion at `world[:address][IDX_FORM]` call
- Handle `lookup_or_generate(world[AudioSource][idx])`
- Implement open universe moves
- Implement OUPM move type
- Automatic conversion at update call for updatespec
- Automatic conversion at update call for extconstaddrs
- Choicemap conversion
- Discard conversion