If we have changed a value in the lookup table before the kernel update,
the lookup table we pass into update should not be equal to the lookup table in
the original trace.

If we do not change any values before the kernel update, but _do_ remove values from the lookup table
after this update, we should pass in an equal lookup table, because from the perspective
of the kernel, the passed-in lookup table in fact did have all these values!

In fact, even if new values were generated before the kernel update, it is reasonable
to have the object be equivalent, since from the perspective of the kernel, the value
could have been in the original lookup table and was just never looked at.


There are many cases where it is totally up to us whether we say two memoized gen fns are equal or not.
If a idx's value changes, then we must consider the new lookup table to no longer be equal.
But if we add or remove indices, it is fine to either consider them equal or nonequal.
If we consider them equal, it matches the case where we had passed in a complete lookup table
to the kernel initially which had all values, and pass this in again.
If we consider them nonequal, it's as though some of the values were there the first time but
not the second, or vice versa.
In fact, even if we don't change any values, it is okay for a lookup table to be considered
not equal to an old one, since this could correspond to the case that the new lookup table
has new indices in it.


To minimize the amount of updating that has to happen, we want to err on the side of considering things
equal whenever possible.  So we will change the equality of a lookup table if we need to change a value in it,
but only then.

### How to "change equality" of lookup tables
Every time we do an update, I think it makes sense to create a shallow copy of each MGF.
(This should not be extremely expensive, as the majority of the "bulk" of the data structure
is in `PersistentHashMap`s, which we only need to copy a pointer to.)
If after the kernel update, we realize we need to delete indices in the a MGF, we will need
to (for the sake of immutability) create some sort of copy of the original MGF, and we will want
this copy to be the one in the updated kernel trace.  So it makes sense to just create these copies for everything
before we run the kernel `update`.  We pass in associated argdiffs which specify that no changes have occurred
yet.  The `lookup` update will be able to dispatch on these and immediately return a `NoChange` if needed.


### "Index Recursion" ie. putting a memoized gen fn in the index

Say we try to implement a sort of recursive function by nesting a memoized generative function
in the index of a lookup.  The generate should work fine.  If we try to `update` a value deep
in the recursive call hierarchy, we do this initial update of this value before
we update the kernel.  Then when we update the kernel, the memoized gen fn we pass in
is not considered equal to the new one.  So the index of the top-level lookup, which
contains this memoized gen fn in it, is now a different index.  Thus the original
index is discarded, and a new value is sampled for this new index.  This is perfectly reasonable behavior,
and maintains correctness.  That said, it means that this memoization does not provide a neat way to
implement incremental `update` for recursion.

In general, using these memoized gen fns in indices is limited in usefulness, anyway,
since there's no easy way to provide constraints for values sampled in the memoized gen function
for these indices, since the memoized gen function object is only created during the generate or update
function call, so there's no way it could be included in a choicemap passed into one of these.
(This isn't a new type of issue.  In the dynamic DSL one could easily construct a mutable object
and use this as an address in the program, and there would be no way to constrain decisions
at this address since the mutable object is only created during the "generate" or "update" call.)


### Providing an explicit recursion mechanism

To allow for recursive behavior, we can update the call signature to `lookup` from
```julia
lookup(lt::LookupTable, idx)
```
to
```julia
lookup(lt::LookupTable, idx, recursive_lookup_table_deps::Vararg{LookupTable})
```
This last `recursive_lookup_table_deps` is used to pass in lookup tables which can be used to
generate values from an underlying generative function.

From the `kernel`'s perspective, this last argument does nothing.  (Ie. if we passed in `lt` as a `ConcreteLookupTable`,
we simply look up the idx from this, regardless of what the `recursive_lookup_table_deps` are.)

However, if we pass in a `lt` which is a `MemoizedGenFn` (with underlying generative function `fn`),
we expect `fn` to have signature
```julia
@gen function fn(idx, recursive_lookup_table_deps...)
```
and when `lookup` generates values by calling `fn`, it passes in the recursive lookup table dependancies.
This allows us to do recursive lookups so that the index does not change just because the
lookup tables needed for a recursive definition change.