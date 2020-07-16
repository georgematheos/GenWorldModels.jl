# Low level implementation for open universe types and updates

### Tell the model the open-universe types & the latents

#### Declare OUPM types

At the beginning of the model, we declare the open universe types as Julia types.
Later we can add syntactic sugar, but the underlying representation is that
each such object stores an `identifier`:
```julia
struct AudioSource <: OUPMType
  id::UUID
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

### Construct OUPMType objects
To construct an OUPMType object, a user should use the syntax
```julia
AudioSource(world, idx)
```
This will construct an `AudioSource` object by looking up the identifier
associated with `idx` in the given `world` object.
If this `idx` is not currently associated, the world
will generate an identifer and fill in the entry in its lookup table.

### Get the index associated with an OUPM object
In a generative function, we can get the index associated with
an object like `source::AudioSource` by calling
```julia
idx ~ lookup_or_generate(world[:index][source])
```

To support this internally, we will augment the `Calls` object to
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
    moves::Tuple{Vararg{OUPMMove}}
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