# Open Universe Types in `UsingWorld`

I would like to support use of types similar to those in BLOG.
A type is essentially a struct with an index and some number of 
"origin" fields which either store another OUType, or Nothing:

```julia
struct SampleOUType <: OUType
    idx::Int
    origin1::Union{Origin1Type, Nothing}
    origin2::Union{Origin2Type, Nothing}
    # possibly include more origin fields...
end
```

The `UsingWorld` combinator can be told to track an open universe type
by specifying "number statements", ie. specifying which of the memoized
generative functions in the world give the number of objects of a given `OUType`
for a specific origin.
For instance, in the classic "aircraft/time/blip" example, we might have

```julia
struct Aircraft <: OUType
    idx::Int
end

struct Time <: OUType
    idx::Int
end

struct Blip <: OUType
    idx::Int
    aircraft::Aircraft
    time::Time
end

UsingWorld(kernel, [memoized_gen_function_specs...], [
    OUTypeSpec(Aircraft, () => :num_aircrafts),
    OUTypeSpec(Time, () => :num_timesteps),
    OUTypeSpec(Blip, (Aircraft, Time) => :num_blips_for_aircraft_at_time, (Nothing, Time) => :num_false_alarm_blips_at_time)
])
```

These `OUTypes` will be useful for automating "split/merge" inference
and sampling objects which have origins of this sort.
In particular, one thing I want to support is a generative
function `sample_with_replacement(world, type, num_to_sample)`
which samples uniformly from among the objects of type `type`.
(Later if needed we could add support for a variation of this
which samples according to some (perhaps unnormalized) distribution
over all the objects.)

To implement `sample_with_replacement` reasonably efficiently, I will
describe 2 different ways to identify objects.  For concreteness,
let's say we have the following types:
```julia
struct U <: OUType
    idx::Int
end
struct T <: OUType
    idx::Int
    u::U
end
```
Say we have 3 `U` objects: `U(1), U(2), U(3)`,
and for each of these we have 1 T objects:
`T(1, U(1)), T(2, U(1)), T(1, U(2)), ...`.
The "official" indexing scheme for `T` objects
is illustrated here: we have the exact origin for the object,
and then the index of this `T` object among all `T` objects with this origin.
But another useful indexing scheme would simply map an integer to a `T` object
in the order we get by iterating over `T` objects:

1 -> T(1, U(1))
2 -> T(2, U(1))
3 -> T(1, U(2))
4 -> T(2, U(2))
5 -> T(1, U(3))
6 -> T(2, U(3))

If `T` depends on 2 origin objects, `A` and `B`, we iterate as follows:
T(1, A(1), B(1)), T(2, A(1), B(1)), T(1, A(1), B(2)), T(2, A(1), B(2)),
T(1, A(2), B(1)), T(2, A(2), B(1)), T(1, A(2), B(2))...

(So we increment the index first, then the rightmost origin, then the
second-to-rightmost-origin, and so on until the leftmost origin.)

To support `sample_with_replacement`, for each `OUType` tracked by the world,
the world will add a generative function `_total_num_*type*`.  This function will
effectively call `total_num_*origin_type*` for every origin type for the type,
and then figure out how many objects for that origin there are, and sum these up.

The world will also expose a deterministic function
```julia
object::outype = object_for_absolute_idx(world::World, outype::Type{<:OUType}, index::Int)
```
which retrieves the object referred to by the "absolute" integer index from the iteration
scheme described above.

`sample_with_replacement`, then, is quite simple: it calls "total_num_*type*",
then makes a call to `uniform_discrete(1, total_num_*type*)` for each object it's going to sample,
and then retrieves the object for the absolute indices it chose.

(In fact, we will have a function `objects_for_absolute_indices` which retrieves the object
for multiple indices at once, since I think it can be more efficient this way.)

For the first draft of this, I think it is probably sufficient to make `sample_with_replacement` a
static generative function.  In future iterations, to support efficient "split/merge" inference,
I will probably need to make this a custom generative function.  (Or maybe not...perhaps
we can just have `objects_for_absolute_indices` understand argdiffs!)