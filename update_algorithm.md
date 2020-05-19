# Update algorithm

Update a `UsingWorld` model is a bit complicated since we need to be able to account for any changes to the topological
ordering of the calls.  This file contains a description of the update algorithm in English+pseudocode.

This algorithm is inspired by David Pearce and Paul Kelly's paper in the "ACM Journal of Experimental
Algorithmics, Vol 11, Artical No 1.7, 2006, Pages 1-24", titled "A Dynamic Topological Sort Algorithm
for Directed Acyclic Graphs", and extends upon the algorithm they present there (the extension is to make
it more efficient in the case we do not require our algorithm to be "unit change", but instead
have access to some information about multiple updates which will need to be performed at once).

### Top-level algorithm
1. Perform all needed updates for `:world` values.
2. Perform the update for the kernel, using the diff for the world.
3. Update lookup counts using the discards from the kernel and all the world updates.

### `:world` update algorithm description

The top of the algorithm is the algorithm we would use if the topological order could not change:
we load all calls we are given constraints for into a priority queue, ordered by the current topological sort,
and update these calls in this order.  If a call has the return value change, we need to add all calls which depend on this
to the priority queue so we can find the weight induced by changing this input to the call.

However, sometimes an "edge" will be added that requires us to change the topological sort.  More specifically,
this means that before the update, we did not need a value for `v` to exist before generating `u`, but after this update,
we need `v` to have a value before we can generate `u`.  This is induced by a call to `Gen.update(lookup_or_generate_trace_for_call_w, new_call=u)`
or `Gen.generate(lookup_or_generate, u)` when running `Gen.update(u)`.
If this occurs, we have to stop what we are doing, and update/generate this new value.  As we do this, we add the calls we are
currently in the process of updating to an `update_stack`.  If we ever try to look up the value of a call on the `update_stack`,
this is an error, since it means that there is a cycle in the new dependencies.

As we update the values in this recursive way, we track the post-order we update them in.  This is the new topological order for this set
of vertices in the graph.  We track these in `call_update_order` in the pseudocode below.  As we update these edges recursively,
I also track the sort of "range" in the topological order of the indices relevant to this update, ie. the highest and lowest indices
in the current topological sort of the calls which are updated.  I call these `fringe_bottom` and `fringe_top`.
To minimize the number of vertices whose topological indices
must change, I do something similar to Pearce and Kelly and track a sort of minimum set of indices which can have calls changed for them.
This is done in `updated_sort_indices`, which contain every ancestor of any edge added during the cycle which is
in the range `[fringe_bottom, fringe_top]` and every every descendent of any edge added during this cycle which is in this range.
To track all these descendants, we sometimes will add a call to the priority queue which we have no argdiff or constraint for,
just to add its index to `updated_sort_indices` and queue up its descendants which have index before `fringe_top`.
Once we exit a recursive cycle, we sort all the indices in `updated_sort_indices`, and reassign these indices to the
calls in `call_update_order` in the order they should now be in.


### `:world` update algorithm pseudocode
1. Load all the calls which update constraints have been provided for
into a priority queue `Q`, with priorities given by the world's current topological sort.
2. Initialize 2 pointers: `fringe_bottom = 0` and `fringe_top = 0`.
INVARIANT: `fringe_bottom` will store a number such that every call whose topological index
is less than `fringe_bottom` has been sorted into the correct position for it to be in after this update;
`fringe_top` stores a number such that any call with topological index greater than `fringe_top` has not yet
had any update performed for it.
1. Initialize a diff, `diff`, of a special type which knows `fringe_bottom`, `fringe_top`, and contains the diff
for every call we have actually run an update for.
The implementation of `Gen.update` will be such that this diff will communicate that:
- Every call before (exclusive) `fringe_bottom` either has the diff `diff` is storing for it, or has no diff.
- Every call between `fringe_bottom` and `fringe_top` (including `fringe_bottom` and `fringe_top`) has an unknown diff,
  unless that call has already been updated, in which case it has the diff returned by that update.
- Every call after `fringe_top` has no diff.
2. Initialize a lookup table `visited` for every call such that `visited[call] = false` for every call initially.
3. Initialize an empty stack `update_stack`.  This will contain every update which is currently in progress.
4. Initialize an empty min priority queue `updated_sort_indices` [to contain all indices updated in an update cycle;
   as a min PQ it will automatically sort these in _ascending_ order]
5. Initialize an empty queue `call_update_order` [to contain the new order an update cycle should now appear in]
6. While `Q` is not empty:
   1. `u = pop!(Q)`
   2. IF `sort[u] > fringe_top`:
      _This means that we have just finished updating a full "update cycle", which we should now sort into the correct order._
      1. Call procedure `reorder_update_cycle!`
      2. `fringe_top = sort[u]`
   3. `fringe_bottom = sort[u]`
   4. `update(u)`
7. Call procedure `reorder_update_cycle!`.

Procedure `update(u)`:
1. IF `u` is on the `update_stack`
   1. ERROR: cycle detected: this update would result in the calculation of call `u` requiring knowledge of the value for call `u`
2. Add `sort[u]` to `updated_sort_indices`.
3. IF there are constraints or is an argdiff for `u`:
   1. PUSH! `u` onto the `update_stack`
   2. Call `Gen.update(u, diff, constraints[u])` if there is a value for `u` in the world; else call `Gen.generate(u, constraints[u])`;
      this returns an `argdiff` which we add to `diff` and a weight which we accumulate.
   3. POP! `u` off of the `update_stack`
   4. PUSH! `u` onto `call_update_order`.
4. Set `visited[u] = true`.
5. IF the `argdiff` is not `NoChange`, THEN add every call which depends on `u` to `Q`.
6. ELSEIF the `argdiff` is `NoChange`, THEN add every call which depends on `u` and is before `fringe_top` to `Q`.
7. RETURN the `argdiff` (possibly to be used in `Gen.update` or generate or something.)

Procedure `Gen.update(u, diff, constraints[u])` / `Gen.generate(u, constraints[u])`
This will run a regular Gen update or generate for call `u` where `diff` denotes
what I articulated above.  During this update,
it is possible that the topological order will be changed.
This can occur in a few ways:

- An edge may be added via a call to `generate(lookup_or_generate,v)`
- One edge may be deleted and another added via a call to `update(lookup_or_generate_trace,v,)`
  where `lookup_or_generate_trace` previously contained a call to `w`.

If either of these occur, adding the dependency that `v` must be generated before `u`, run the following procedure:

Procedure `generate(lookup_or_generate, v)` OR `update(lookup_or_generate_trace, v)`:
1. Increment the `lookup_counts` for how many times `v` is looked up in `u`. (`u` is the top value on the `update_stack`).
2. IF `sort[v] < fringe_bottom`
   _In this case, we have already updated v if we need to, so no new dependency is added._
   1. Set `df = NoChange()`.
   ELSEIF `visited[v]`
   _In this case, `v` is between `fringe_bottom` and `fringe_top`, and has already been updated, so we have a diff articulating what new value `v` will take._
   1. Set `df = diff[v]`
   ELSE
   _In this case, `v` is after `fringe_bottom` and has not been updated! We need to run an update on it before the algorithm can continue._
   1. Set `fringe_top = sort[v]`
   2. Set `df = update(v)` (Ie. call procedure `update` on `v`.)
3. This call to `update(lookup_or_generate_trace)` should RETURN the new trace implied by applying the diff `df` to the previous value for `v` in the world.

Procedure `reorder_update_cycle!`:
_This is called when we just completed the last update on the update stack, and have now moved past the `fringe_top` call, so we can reorder the updates between `fringe_bottom` and `fringe_top` as needed._
1. WHILE `call_update_order` IS NOT empty
   1. `u = pop!(call_update_order)`
   2. `idx = pop!(updated_sort_indices)`
   3. Set `sort[idx] = u`