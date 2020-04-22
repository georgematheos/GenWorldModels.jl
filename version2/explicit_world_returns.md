I've thought this through more and decided this probably isn't the right call,
but here was one idea I'd had for how we could design the combinator.

## Proposal for version 2 of the "UsingMemoized" combinator

Takes as arguments:
- the kernel
- some number of `MemoizedGenFn` objects

A `MemoizedGenFn` object is immutable, and stores:
- a pointer to a generative function
- a lookup table of all the "generated" values

The lookup table will be implemented as a `PersistentHashMap` or something similar,
so that as we add values to the `MemoizedGenFn` we create new `MemoizedGenFn`
objects.

So we can define
```julia
struct MemoizedGenFn
    gen_fn::GenerativeFunction
    values::PersistentHashMap
end
```

We will have a generative function
```julia
(value, new_mgf) ~ lookup_or_generate(mgf::MemoizedGenFn, key::Any)
```
which will check if the key is in `keys(mgf.values)`.  If it is,
it returns this key with probability 1; otherwise, it will
generate the value by calling `mgf.gen_fn(key)`.  In either case,
`lookup_or_generate` will return the tuple `(value, new_mgf)`
where `value` is the "return value", and `new_mgf` is a `MemoizedGenFn`
which is the same as the original `mgf`, except now contains a value for
`key` if no value was provided in `mgf`.

As syntactic sugar, to avoid typing `lookup_or_generate`, I propose that we could
assign `⋉ = lookup_or_generate`.  `⋉` is a unicode character which supports infix notation,
so we could write code along the lines of

```julia
value, new_mgf ~ mgf⋉(key)
```

This lets us do our lookups in a syntax pretty close to a standard generative function call;
instead of `mgf(key)` we type `mgf⋉(key)`. 
There's no reason in particular to use `⋉`; a full list of characters which support infix notation
is given at https://github.com/JuliaLang/julia/blob/master/src/julia-parser.scm .
Anyway, that's kind of a tangent.

---

The main difference between this and the previous version of `UsingMemoized`
is that the "lookup table" is now an immutable object which the `kernel` is responsible
for keeping track of.  Ie. every time we look up something from the memoized generative function,
a new `mgf` object is returned, and the kernel needs to be able to continue passing this along.
This means that oftentimes when using the `Map` combinator in the old version of `UsingMemoized`,
we will now need to use `Unfold` so the return values can propogate.
What this affords us is that whereas the old version of `UsingMemoized` needed some nasty code to track
the dependencies of looked-up values, now the program graph will keep track of it automatically.

The `UsingMemoized` combinator will essentially now just provide some syntactic sugar by
disentangling the choicemaps for the memoized generative function from the choicemap for the kernel itself.

As an example:
```julia
@gen function generate_size(object_index)
    size ~ normal(object_index, 1)
    return size
end

@gen function kernel(get_size)
    obj1_size, get_size = {:obj1_size} ~ get_size⋉(1)
    obj2_size, get_size = {:obj2_size} ~ get_size⋉(2)
    
    return obj1_size + obj2_size
end

generate_2_sizes = UsingMemoized(kernel, :sizes => generate_size)
```

```julia
@gen (static, using_memoized) function generate_2_sizes()
    obj1_size ~ generate_size[1]
    obj2_size ~ generate_size[2]
    
    return obj1_size + obj2_size
end

# expands to

@gen (static) function generate_2_sizes_kernel(generate_size0)
    obj1_size, generate_size1 = {:obj1_size} ~ generate_size0⋉(1)
    obj2_size, generate_size2 = {:obj2_size} ~ generate_size2⋉(2)
    
    return obj1_size + obj2_size
end

generate_2_sizes = UsingMemoized(generate_2_sizes_kernel, :generate_size => generate_size)
```


```julia
@gen (static, using_memoized) function generate_n_sizes(num_samples, num_objects)
    samples ~ [uniform_discrete(1, num_objects) for i=1:num_samples]
    sizes ~ [generate_size[samples[i]] for i=1:num_samples]
end

# expands to:

@gen (static) function generate_n_sizes_kernel(get_sizes0, num_samples, num_objects)
    samples ~ [uniform_discrete(1, num_objects) for i=1:num_samples]
    sizes, get_sizes1 = {:sizes} ~ lookup_or_generate_for(get_sizes0, samples)
end

generate_n_sizes = UsingMemoized(generate_n_sizes_kernel, :generate_size => generate_size)
```

where we have previously defined

```julia
@gen function folded_lookup_or_generate(i, mgf, keys)
    return @trace(lookup_or_generate(mgf, keys[i]))
end

@gen (static) function lookup_or_generate_for(mgf, keys)
    vals_and_mgfs ~ Unfold(folded_lookup_or_generate)(length(keys), mgf, keys)
    vals = map((val, mgf) -> val, vals_and_mgfs)
    new_mgf = vals_and_mgfs[end][2]
    return (vals, new_mgf)
end
```


***
Update: I thought about it more and realize that it probably won't be possible to make
that very performant, since if we remove a key or something we will have to propogate
the update through every time the mgf value is used.

However, I think that understanding memoized generative functions as being equivalent to
this sort of thing is probably a good formalism.

In general one thing we will want is a way to have a static generative function not try
to update every use of an arg with a diff if the argdiff is a indexdiff.

Idea:

```julia
abstract type IndexDiff{T} <: Gen.Diff end
struct GeneralIndexDiff{T} <:  IndexDiff{T}
    changed_indices
end
struct SingleIndexDiff{T} <: IndexDiff{T}
    idx::T
end
idx_changed(diff::GeneralIndexDiff, idx) = idx in diff.changed_indices
idx_changed(diff::SingleIndexDiff{T}, idx::T) where {T} = idx == diff.idx

@gen (static) function test(a)
    b ~ uniform_discrete(1, 5)
    x ~ normal(a[1], 1)
    y = x + a[b]
    return y
end
```

The static DSL will be able to understand that if we do an update with the argdiff for `a` being an `IndexDiff`,
we only actually perform the update for the nodes in the static graph where the lookup index is the index in the which
has a changed value.

***

```julia
@gen function kernel(mgf)
    val ~ lookup_or_generate(mgf[key])
end
```