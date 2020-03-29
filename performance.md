### Asymptotic performance

Say we do an `update` to a trace for a `UsingMemoized` gen function with `m` 
memoized generative functions, `u` updates to a lookup in an
MGF.

This requires:
- shallow-copy each mgf: `O(m)`
- update each mgf value from `constraints`, and downstream dependencies effected by this
    - to do this, we must sort these constraints, which is `O(u*log(u))`
        - we only achieve this worst case if all updates are known at once
        (the opposite would be if we just have a chain of dependent updates, so 
        at any time we only know the next one we need; then sorting is `O(1)`)
        - we must also update all the values, or generate new ones we now need
- update the kernel
- update the lookup counts & drop unused calls
    - currently this requires a full scan of the `discard` ultimately returned 
    by the `UsingMemoized` generative function,
    which is `O(|V|)` where `|V|` is the number of nodes in the discard graph.
- update the topological sort
    - currently I just resort, which takes `O(number of mgf lookups)`
    - it looks like algorithms exist to update topological sorts which
    are `O(d * log(d))` where `d` is the distance between `x` and `y`,
    where `x --> y` is an edge being added to our graph which invalidates
    the current topological sort.

So currently the runtime is:
`O(m + u*log(u) + |V| + num mgf lookups)`
where
- `m` = number of mgfs
- `u` = number mgf lookups to update
- `|V|` = number nodes in the returned discard graph

If we use the dynamic topological sort algorithm,
`O(m + u*log(u) + d*log(d) + |V|)` would be
the runtime.  The `d * log(d)` term would probably be small for most
practical applications where we don't drastically alter
the dependency graph during inference.

How to get rid of the `O(m)` and `O(|V|)`?

So long as we require traces to be immutable and have to allow for dynamic
restructuring of the dependency graph, we can't remove the `O(m)`, since
we have to make a shallow copy of every `mgf` in case it is modified
during a change to the dependency graph.  However, if we had mutable
traces, we would not need to copy these, and could get rid of this term.
Likewise, if we could statically guarantee for some updates
that only some subset of the `mgf`s will be modified in any way,
we will only need to copy these.

To get rid of the `O(|V|)` we would need some way to maintain counts of which
subtraces use lookups to different addresses dynamically, so we don't have to
count these up at the end for each update.  One way to do this would
be to implement a special type of choicemap which is given some functions
to accumulate values as it is generated.  The idea would be that
if we could specify for the static DSL to use such choicemaps,
then as it `generate`s its trace, every branch knows the counts,
so when it discards branches later, we don't need to recalculate these values.
This would not be an asymptotic improvement for the combined procedure of
generating then later updating traces, since we have to at some point generate
every branch in every choicemap, so scanning again when we discard just doubles
our work.  But still, it could halve the work we have to do!