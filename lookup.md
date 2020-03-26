# Is `lookup` a generative function?

It seems problematic to call `lookup` a generative function because it _does_ mutate the external state.
However, I will try to articulate why I think it is valid to consider `lookup` a generative function.
The short version is: when `lookup` is called on a valid set of parameters--a lookup table with every value already filled in--
it does not mutate the external state.  However, in `UsingMemoized`, we use `lookup` in a formally invalid way, since we 
pass in objects which `lookup` does mutate.  In this specific case, we make sure everything works out so that the top-level
`UsingMemoized` is still a valid generative function.
If users try to use `lookup` in a mutating way elsewhere, though,
this is invalid, since they are passing in arguments not formally in `lookup`'s domain.
While it seems problematic to expose a generative function which can be misused in this way,
Gen already exposes such generative functions (eg. `Map`), so this is not a new problem;
it is simply part of Gen's architecture that users must make sure they are passing in
arguments to generative functions which are within these functions' domains.

### Non-mutation is not enforced by the Gen system, but by the user

One of the requirements in the formal definition of a generative function is that it cannot mutate the externally
observable state.  However, in practice this definition is difficult to enforce, as a generative function
can often be misused in a way to cause external state to be mutated.

For instance, one reasonable-looking generative function is `sample_equal_std_normals`,
defined as follows:
```julia
@gen (static, diffs) function sample_normal(mean, std)
    val ~ normal(mean, std)
    return val
end
sample_normals = Map(sample_normal)

@gen (static, diffs) function sample_equal_std_normals(means::AbstractVector, std::Float64)
    num = length(means)
    vals ~ sample_normals(means, fill(std, num))
    return vals
end
```

This appears to be totally valid.  But I could create the following pathological case:
```julia
mutable struct TrackLengthCallsVector <: AbstractVector
    v::Vector
    num_length_calls::Int
end
TrackLengthCallsVector(v::Vector) = TrackLengthCallsVector(v, 0)

# when we call length, we mutate the `t` object!
function Base.length(t::TrackLengthCallsVector)
    t.num_length_calls += 1
    return length(t.v)
end

# ...
# define getindex, etc. for `TrackLengthCallsVector`
# ...

vec = TrackLengthCallsVector([1, 2, 3])
println(vec.num_length_calls) # prints "0"
normals = sample_equal_std_normals(vec, 1)
println(vec.num_length_calls) # prints "2" (one `length` call in sample_equal_std_normals, another in the `Map` combinator)

```
Here, by overriding the `length` method for a type, it turns out `sample_equal_std_normals`
_can_ modify the external state.

In general, to prevent this sort of thing from happening, we could add the restriction
that no mutable objects can be used as arguments to generative functions.  This seems pretty restrictive,
practically speaking, so it seems that a better formalism is simply to say that
any generative function implicitly defines a domain for its arguments.
In the case of `sample_equal_std_normals`, this domain is the set of tuples
`(means, std)` where `std` is any `Float64`, and `means` is any `AbstractVector`
which is not mutated by calls to `length` or `getindex` (which is used in `Map`).
The Gen system does not enforce this domain--ie. it does not throw an error if the user passes in
a `means` object which is mutated--but formally, `sample_equal_std_normals` is not a generative
function for this input.  Just as Gen does not enforce that the inside of generative functions
written in the static and dynamic DSL do not mutate external state, and it is up to users to make
sure functions are written this way, it is up to users to make sure generative functions
are called with arguments which are in the generative functions' domains.

Notably, even if we did not call `length` in our static function `sample_equal_std_normals`,
since `length` is called in the `Map` combinator, this issue would still occur.
So as is, Gen exposes non-user-defined generative functions which can be misused, and
which it is up to the user to use correctly.

### Lookup
Lookup is used as follows:

```julia
looked_up_val ~ lookup(lookup_table::LookupTable, idx)
```

Formally, the domain of `lookup` is tuples `(lookup_table, idx)` where `lookup_table`
contains a value for key `idx`, such that the call `lookup(lookup_table, idx)`
does not mutate `lookup_table` (or `idx`).  When used "properly" in this way,
the function returns the value for `idx` stored in `lookup_table` with `logpdf = 0.`.
If a user tries uses `lookup` on a tuple `(lookup_table, idx)` where
`getindex(lookup_table, idx)` mutates `lookup_table`, this is considered the same sort of user
error as if user tries to call `Map(gen_fn)(vect)` where `length(vect)` mutates `vect`.

### `UzingMemoized`

The `UsingMemoized` combinator takes as its first argument a generative function `kernel`.
This `kernel` accepts as it's first argument a `LookupTable` whose values can be accessed
by the `lookup` function.  It should assume this lookup table contains a value for every `idx`
it would like to look up in it; the `UsingMemoized` function makes the promise that the lookup table
will behave identically to if this were the case.

In reality, the `LookupTable` passed into `kernel` will not be pre-populated with all the needed values.
However, the behavior from the perspective of `kernel` will be identical to if this were the case.
Thus technically the `UsingMemoized` combinator does not use `kernel` as a generative function,
since it uses `kernel` in such a way that `kernel` can mutate state observable external to `kernel`.
However, it does this in a very controlled way.  It knows `kernel` cannot mutate externally observable
state if it is used correctly, ie. if the lookup-table is pre-populated.  Therefore it knows that the only instance in which
mutation can occur is then when `lookup` mutates the lookup table, since this is the only method we have
guaranteed to the kernel will behave as though it does not mutate the lookup table.

(We assume that the only way users lookup values
from the lookup table is via the `lookup` generative function, as this is the only "public" interface via
which these objects may be indexed into.  However, unlike Java, which makes it possible to programmatically enforce
such interfaces by throwing errors, it is impossible in Julia to programmatically prevent users from using
the internal interface of the lookup-table.  Internally, there will have to be some way for `lookup`
to get values out of the data structure, and presumably a user _could_ read the source code
and figure this out.  But we make no guarantees to users about the behavior if they do this,
so for the users to ensure that their `kernel` does not mutate state, since the only guarantee we make about the
lookup tables is that they will behave as though `lookup(lookup_table, idx)` does not mutate them,
to ensure they are obeying the GFI by not mutating external state, they cannot use `lookup_table`s via the private
interface.)


#### Kernel purity requirement
Note that this guarantee of identical behavior assumes that the `kernel` function is a pure
function, ie. the distribution it defines over traces is independent of the global program environment
given its inputs.  (Or, at least, independent of the global program environment given its inputs
and the current state of the global random number generator.)  The reason we need this restriction
is that if the generative function can access the global timer, it might be able to detect a difference
in how long it takes to look things up in the LookupTable the first vs second time an index is accessed.
But this requirement of `purity` seems to me something which should be included as part of the definition
of a generative function anyway.