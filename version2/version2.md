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
world_spec::WorldSpec # specifies all the MGFs, etc; created in the `UsingMemoized` constructor

world[:addr]::MemoizedGenerativeFunction
world[:addr][key]::MemoizedGenerativeFunctionCall

# looks up the given key for the mgf in the world; generates the value if doen't yet exist
function lookup_or_generate(::MemoizedGenerativeFunctionCall) end

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
The world object will
- Create a dictionary from lookup -> constraint
- For every lookup constrained in `constraints[:world]`, load the lookup into a heap, with priorities given by the topological sort on lookups
- Create an empty `IndexDiff`, `diff`
- Create a set/list of all the new "forward-edges" for the dependency graph, ie. new dependencies which will exist
after this update, but did not beforehand, and a list for all deleted edges.
- WHILE the heap is not empty:
  - Pull the highest-priority lookup off the heap.  Call this `l1`, with position `i` in the topological ordering.
  - Get the constraints for `l1` from the dict
  - Note in the world state that we are running an update on `l1`
  - Call `update` on the trace for lookup `l1`, passing in the `diff` for the world, and the constraints for this lookup
  - IF, during this update, a `lookup_or_generate` generate is triggered:
    - Note that this generate was called from `l1` in our "new forward-edge list", and generate as usual.
  - IF, during this update, a `lookup_or_generate` update is triggered for lookup `l2` is triggered:
    - If `l2` is not in the topological ordering, or is index `j` in the topological ordering
    with `j < i`, note that this lookup was called from `l1` in our "new forward-edge list", and update as usual.
    - If instead `j > i`, it means that this update is altering our dependency graph in such a way that our current topological order
    will become invalid.  Either this update causes a cycle, in which case it is invalid, or we will be able to create a new topological
    ordering.  We will proceed as though it will eventually work out.
      - Halt this update.  Pull the next item off the heap, and proceed with updating it, and so on, until we either reach an item
      with topological index `>= j` or we reach an item with `l1` as an ancestor in the dependency graph.
        - If we have reached index `>= j`, resume our update for `l2`, then for `l1`.  Since the `diff` object is being mutated as we
        proceed, these updates will complete with the `diff` passed into them having information from all the updates with indices between `i` and `j`.
        Note in our "new forward-edge list" that we now have dependency that `l2` is looked up in `l1`.
        - If we did not reach index `j`, but instead reached an update for `l3` which has `l2` as an ancestor in the dependency graph,
        proceed with updating `l1`, then `l2`.  Note in our "new forward-edge list" that we now have dependency that `l2` is looked up in `l1`.
  - Once the update for `l1` is complete, scan the discard for this update to see what lookups are no longer performed for `l1`.
  Update our dependency graph by subtracting these lookup counts from those in the graph.  If we "delete an edge" in this graph,
  add the deleted edge to our list.

- Once the heap is empty, all these updates have been performed.  Update the topological sort using the new and deleted edges
in our lists.  (There are known algorithms for how to efficiently update a topological sort.)  If this sort is impossible,
it means the update is invalid, so return an error.  Note that all this work involving topological sorting is somewhat computationally expensive,
but in practice most updates will not induce structural changes.
- Note that while these updates are occurring, the world object should be accumulating the total weight of all the updates,
updating the "score" of the world, and putting together a discard.
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

I think it *would* be possible to implement `UsingWorld` so that we don't have restriction 2, by building our dependency
graph by looking at the choicemaps from generating instead of tracking calls to `generate`, `update`, and `regenerate`.
However, this would add a bit of complexity, and would reduce performance.  (For example, when running updates,
if we had to scan the new choicemaps and detect differences from the old ones to see how the dependency graph has updated,
this would be very slow, compared to just observing what new `update` and `generate` calls occur!)
However, at some point I could add support for this "safe" mode, and make it some sort of
kwarg option to turn off the safe mode and do these performant updates which rely on restriction 2.