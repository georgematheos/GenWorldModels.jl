# Diff propagation for Memoized Generative Function Calls

I'm not totally sure we should actually use argdiff propagation for most of the information about
how `lookup_or_generate` calls are changing; I'm going to try to work through it here.

## Updating a `lookup_or_generate` trace

When we encounter a `lookup_or_generate` call during an update, we can be in one of a few situations:
1. The key to the MGF has changed
2. The world has changed in such a way that the value for the MGF lookup has changed
3. The world is in the process of changing in such a way that the value for this MGF lookup *may* change, though
   it has not been updated yet.
4. We can be sure that during this update, the value for this MGF lookup has not changed.

When there are automatic conversions between concrete and abstract forms, these more specific things can also occur:
1. During a "get abstract" call, the abstract origin of the passed-in concrete object has changed in the world
2. During a MGF call with a concrete object, the associated abstract object has changed

These cases add some "intermediate steps" we may care about.

### Argdiffs?

It seems very easy to use argdiffs to convey when the key changes, since this will occur when a
key diffed with `UnknownChange` is looked up in an MGF.  (Ie. `world[addr][Diffed(key, UnknownChange())])`.)

A `world` which has some changed values should at least be diffed with a special diff type, say `WorldUpdateDiff`,
to communicate that we can get information about it by looking at it's current update state.  We can probably
have `world[addr]` be diffed with the same type.

If a key has not changed, we could be in any of the possibilities 2-4.  If we're in possibility 2 (value change),
we may have noted a diff in the world we want to access and return.  Another consideration is that we should try to minimize
the amount we put in the discard (while retaining correctness) to speed up the discard scanning.

I want to avoid the overhead of calling `lookup_or_generate!` when possible (this is probably a low overhead, but at the margin
it adds a call to `has_val`), so the question mostly comes down to whether to have if statements in the `lookup_or_generate` `update` function,
or in some diff propagation code.  The advantage of putting it in the diff propagation is that this means the implementations
of auto concrete-->abstract and auto concrete-->abstract origin in the static DSL can also utilize the diff propagation behavior
and hence update the minimal number of `lookup_or_generate` calls.

### Argdiff TODO once we have some static analysis

We could possibly automatically have `world[addr]` return a `NoChange` if we can be sure at some point that no MGFs for a given address
can be updated.  For this, we will need some static program analysis so we can be sure that we have already passed the point
where we might have to update any `world[addr]` calls.  To do this, it would suffice to statically analyse the DSL functions making up the MGF
to form a directed graph of which MGFs can look up which others; we could then find the strongly connected components of this graph
and use this as part of the ordering.  If we are updating a MGF call, we can be sure any MGF call in an "earlier" SCC has either been updated
or will not be updated.

## Argdiff behavior

1. A world being updated is diffed with a `WorldUpdateDiff`.
2. A `world[:addr]` may also be updated with a `WorldUpdateDiff` if the underlying world has this diff.
3. `world[:addr][key]`, where `world[:addr]` has a `WorldUpdateDiff`, obeys the following logic:
   1. If the key has changed, we use a `KeyChangedDiff`
   2. Else, IF this is a simple MGF call, so `key` is not a concrete OUPM object, and `addr` is not special:
      1. If the update state is past the point of having updated this call:
         1. If there is a diff for the call in the world, have a `ValChangedDiff`
         2. Else, `NoChange`
      2. Else, `ToBeUpdatedDiff`
   3. IF this is a MGF call to `get_abstract`:
      1. If there is no change to the association, `NoChange` (=if there is a value, and there is no diff in the lookup table)
      2. If there is a change, `ValChangedDiff` (in which case there MUST be a diff in the lookup table)
      3. If there is no longer an association, `ToBeUpdatedDiff`

The automatic conversions are negotiated via calls to "get abstract", so I don't think we need any special-case behavior for these to work.

Note that I think all update calls involving a keychange could be implemented simply by generating a new trace and discarding the old one,
then returning `UnknownChange()` or `NoChange()` depending whether the retvals of the old and new traces match.