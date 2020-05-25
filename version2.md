New version's syntax will combine all the "memoized generative functions" into a single world object.

Will look something like:
```julia
@gen function approx_fib_helper(world, n)
    if n == 0 || n == 1
        val ~ normal(1, 0.1)
    else
        fib_n_minus_1 ~ lookup_or_generate(world[:approx_fib][n - 1])
        fib_n_minus_2 ~ lookup_or_generate(world[:approx_fib][n - 2])
        val ~ normal(fib_n_minus_1 + fib_n_minus_2)
    end
    return val
end

@gen function approx_fib_kernel(world, n)
    val ~ lookup_or_generate(world[:approx_fib][n])
    return val
end


const approx_fib = UsingWorld(approx_fib_kernel, :approx_fib => approx_fib_helper)

# use:
tr, _ = generate(approx_fib, (5,))
```

We have a few options for the choicemap.

By default, the choicemap will look something like:
```
:world
- :approx_fib
  - 5
    - :val - 5.1
  - 4
      - :val - 3.01
  - 3
      - :val - 2.91
  - 2
      - :val - 2.09
  - 1
    - :val - 1.04
  - 0
    - :val - 0.84
```
ie. we totally hide all the lookups.

If we want to see the lookups,
we can instead create: `UsingWorld(... ; show_lookups_in_choicemap=true)`.  Then the choicemap is:
```
:kernel
  - :val
- 5.1

:world
 - :approx_fib
  - 5
    - :fib_n_minus_1 - 3.01
    - :fib_n_minus_2 - 2.91
    - :val - 5.1
  - 4
      - :fib_n_minus_1 - 2.91
      - :fib_n_minus_2 - 0.84
      - :val - 3.01
  - 3
      - :fib_n_minus_1 - 0.84
      - :fib_n_minus_2 - 2.09
      - :val - 2.91
  - 2
      - :fib_n_minus_1 - 1.04
      - :fib_n_minus_2 - 0.84
      - :val - 2.09
  - 1
    - :val - 1.04
  - 0
    - :val - 0.84
```

If we don't want to spend any time hiding values in the choicemap, we can use
the option `UsingWorld(...; unmodified_choicemaps=true)`.  This will return 
a choicemap with the same address scheme as the `show_lookups_in_choicemap`
choicemap, but with the addresses for `lookup_or_generate` calls pointing to
sub-choicemaps with some metadata needed for `UsingWorld`.

## Internals

```julia
world::World # passed into the kernel, into the MGFs, etc.; constructed when calling `generate(::UsingMemoized)`; contains pointer to a world_spec

world[:addr]::MemoizedGenerativeFunction
world[:addr][key]::MemoizedGenerativeFunctionCall

# looks up the given key for the mgf in the world; generates the value if doen't yet exist
function lookup_or_generate(::MemoizedGenerativeFunctionCall) end

# for syntax improvements, we could implement syntax like:
function lookup_or_generate(::MemoizedGenerativeFunction, key::Any) end
⋉ = lookup_or_generate
# enables the syntax
val ~ memoized_gen_fn⋉(key)
# though we may not be able to make this as efficient as the mgf[key] syntax
# since we may be able to add special code to the static DSL so `IndexDiffs` are understood for `mgf[key]` indexing syntax
```

### Generate
1. Construct an empty "world" object
2. Give the world object the `constraints[:world]` choicemap.
  The world object will be coded so that when we generate values in the world, if there is a constraint
for that value, the constraints are effected.
3. Tell the "world" object we are going to generate the kernel.
4. Call `generate` on the kernel.
  While we generate, values will be populated in the `world`. The `world` will track the dependencies
  between lookups.  The `world` will also note the topological order these lookups are processed in.
5. Tell the "world" object that the kernel has been generated.
  If the world object detects that there are constraints it has been given for keys which were never generated,
it will throw an error.
6. Calculate the score (using the score stored in the `world`); construct the trace; return.

### Update / Regenerate
1. Create a shallow copy of the world object
2. Give the world object the `constraints[:world]` choicemap and tell it to run the specified updates.
The world object will run the algorithm described in `update_algorithm.md` to update the calls in the world.
3. Tell the world we are going to run a kernel update, then update the kernel using the updated world object and the `diff`.
- If new lookups are `generate`d, the world will update the dependency graph accordingly, and add these new lookups to the end of the topological ordering.
4. Scan the discard for the kernel update, and decrease counts in the world for the lookups which were dropped.  Delete lookups if needed,
and add these to the discard.
5. Sum up weights, scores, and put together the final discard; construct the new trace; return.

For regenerates, the exact same procedure should be followed, except with `Selection`s rather than `ChoiceMap`s being passed around.

### Update performance
Here are several performance issues, and possible fixes:
1. Immutability: A complicated update will involve many `PersistentHashMap` updates and `PersistentVector` updates, which can be slow.
Eventually by allowing for mutable traces we could probably avoid this issue.  (Though if we want to make it possible to construct
"patch" objects, conceptually the performance will be similar, though we may be able to implement it better.)
2. Resorting: if the topological ordering needs to change in a complicated way, it will involve reindexing that will
probably take something like O(n*log(n)), where n is the difference between the highest index in the ordering and lowest
index in the ordering of spots whose position needs to change.
Most of the time, an update will result in very little change in the structure, so this is fine.  We can hardcode some threshold
for `n` above which we will just redo the topological sort from scratch rather than trying to update to try to optimize performance.
3. Overhead for being prepared to resort and remove lookups: if we get everything else to be very fast, having to do all setup
to be prepared to reorder our topological sort, and to drop lookups, will be a significant performance drag on small updates.
To fix this, we could create a custom `UpdateSpec` type which guarantees that the structure of the graph will not change during this update.
We could also probably automatically detect some things (eg. dispatch on the case where we don't have any world updates),
but my sense is that oftentimes, it will be complex enough to detect this that it might be better to simply have this information
provided by the user.
4. Discard scanning is slow
For this, we could update the static modeling language so that we can specify to it to accumulate a certain value on its choicemaps;
then the discard choicemaps will all already have the counts on them, so we won't need to scan everything at the end.
This still adds a bit of overhead, so for updates where we are sure no lookups are going to be dropped, we could
again have a custom `UpdateSpec` so we can specialize to this case.
5. We need `IndexDiff` support - currently, the static DSL will call an update on every use of the `world`, even for lookups on indices which don't change.
We should update the static DSL to keep track of when nodes in the static IR graph are only used via a lookup for a certain index,
and then when an `IndexDiff` appears for that node, we only call an update on the nodes which use either the whole node as input,
or an index from it which is actually updated.

### Restrictions
1. The `world` object may only be used in calls of the form `lookup_or_generate[:gen_fn_addr][key]`.

2. The `UsingWorld` combinator assumes that the choicemaps returned by the kernel and the memoized generative functions
are all precise accounts of how `lookup_or_generate` was used in these functions.  In other words, all `lookup_or_generate`
calls MUST be traced.  When running `generate` on the kernel or a memoized generative function,
every `generate(lookup_or_generate)` call must correspond to exactly 1
choicemap for `lookup_or_generate` which is exposed in the resulting trace.
When running `update` on the kernel or a memoized generative function, every `generate(lookup_or_generate)` call must
correspond precisely to a call to `lookup_or_generate` which did not appear before the update.  Every `update(lookup_or_generate)`
or `regenerate(lookup_or_generate)` call must correspond precisely to updating a `lookup_or_generate` call which did appear before
the update.

The reason for these restrictions is that the `UsingWorld` combinator uses the order of calls to `generate`, `update`, and `regenerate`
to understand which lookups in the world depend on which other lookups, but look at the `discard` choicemaps from updates
to understand when these dependencies are removed.  Therefore if there is any mismatch between the calls to these functions,
and the choicemaps, `UsingWorld` will be unable to keep track of everything precisely, and errors will occur.