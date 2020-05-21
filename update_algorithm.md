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
   2. IF NOT `visited[u]`
      1. IF `sort[u] > fringe_top`:
         _This means that we have just finished updating a full "update cycle", which we should now sort into the correct order._
         1. Call procedure `reorder_update_cycle!`
         2. `fringe_top = sort[u]` _(technically this is unnecessary since we will otherwise do this in `update_or_generate`; I include it for clarity and to keep `fringe_bottom < fringe_top`)_
         3. `fringe_bottom = sort[u]`
      2. `update_or_generate(u)`
7. Call procedure `reorder_update_cycle!`.

Procedure `update_or_generate(u)`:
1. IF `u` is on the `update_stack`
   1. ERROR: cycle detected: this update would result in the calculation of call `u` requiring knowledge of the value for call `u`
2. Add `sort[u]` to `updated_sort_indices`.
3. `fringe_top = max(fringe_top, sort[u])`
4. IF there are constraints or is an argdiff for `u`, or there is no value for `u` in the world:
   1. PUSH! `u` onto the `update_stack`
   2. Call `Gen.update(u, diff, constraints[u])` if there is a value for `u` in the world; else call `Gen.generate(u, constraints[u])`;
      this returns an `argdiff` which we add to `diff` and a weight which we accumulate.
   3. POP! `u` off of the `update_stack`
5. PUSH! `u` onto `call_update_order`.
6. Set `visited[u] = true`.
7. IF the `argdiff` is not `NoChange`, THEN add every call which depends on `u` to `Q`.
8. ELSEIF the `argdiff` is `NoChange`, THEN add every call which depends on `u` and is before `fringe_top` to `Q`.
9. RETURN the `argdiff` (possibly to be used in `Gen.update` or generate or something.)

Procedure `Gen.update(u, diff, constraints[u])` / `Gen.generate(u, constraints[u])`
This will run a regular Gen update or generate for call `u` where `diff` denotes
what I articulated above.  During this update,
it is possible that the topological order will be changed.
This can occur in a few ways:

- An edge may be added via a call to `generate(lookup_or_generate,v)`
- One edge may be deleted and another added via a call to `update(lookup_or_generate_trace,v,)`
  where `lookup_or_generate_trace` previously contained a call to `w`.

If either of these occur, adding the dependency that `v` must be generated before `u`, run the following procedure:

Procedure `Gen.generate(lookup_or_generate, v)` OR `Gen.update(lookup_or_generate_trace, v)`:
1. Increment the `lookup_counts` for how many times `v` is looked up in `u`. (`u` is the top value on the `update_stack`).
2. If `v` has not yet had a value generated for it, add `v` to the end of the topological sort.
3. IF `sort[v] >= fringe_bottom` AND NOT `visited[v]`
   _In this case, call `v` has not been updated yet; we have to run this update before we can proceed._
   _Also, note that this ALWAYS occurs if no value for `v` has been generated yet since then `v` is at the end of the topological sort._
   1. Call `update_or_generate(v)`
4. RETURN the lookup_or_generate trace containing the value stored for `v` in the world.

Procedure `reorder_update_cycle!`:
_This is called when we just completed the last update on the update stack, and have now moved past the `fringe_top` call, so we can reorder the updates between `fringe_bottom` and `fringe_top` as needed._
1. WHILE `call_update_order` IS NOT empty
   1. `u = pop!(call_update_order)`
   2. `idx = pop!(updated_sort_indices)`
   3. Set `sort[idx] = u`

## "Proof of Correctness"
We would like to guarantee that when the algorithm is run:
1. Every call for which a constraint was provided had an `update` or `generate` called for it.
2. Every call which looks up a call whose value changes had an `update` called for it.
3. At the end of the update, the topological order of the vertices is correct for the new dependency
   structure of the graph, which is the old dependency structure, PLUS dependencies coming from
   updates with changed keys and new generate calls, and with dependency removals indicated by
   the discards.

#### Proof of 1 & 2:
Claim: every call which ends up in the queue `Q` has an `update` or `generate` called for it
if there is a constraint or argdiff for it.
Pf: The "`:world` update algorithm`'s step 6 pops every element out of the queue and runs `update_or_generate` on it
if it has not yet been visited.  We only mark calls as visited in `update_or_generate`. This means that
every node which ends up in the `Q` must have `update_or_generate` called on it at some point.
`update_or_generate` always runs `update` or `generate` for the call if there are constraints or argdiffs.
Thus for 1 it suffices to show that every constrained call is put on the `Q`, and for 2 it suffices
to show that every call which depends on a call whose value changes is put on the Q.

At the beginning of the algorithm, all constrained calls are enqueued onto `Q`.  Therefore (1) is guaranteed.

Furthermore, if after an update in `update_or_generate`, a call has an argdiff, all calls dependent on this
are added to the `Q`.  Therefore (2) is guaranteed.

#### Reformalizing (3)

It remains to prove (3).

Let me formalize this third proof goal a bit more clearly.
Before the update, there is a topological order for the dependency graph.
In the process of updating the constrained calls (and those downstream from
these), a call may have a dependency removed (which does not invalidate
the topological order), or may have a dependency added.
We assume that when the dependency that `u` looks up `v` is added,
this is indicated as follows: when running `Gen.update` or `Gen.generate`
on `u` during the world update, one of the following calls is made:
- `Gen.update(lookup_or_generate_trace, old_call=w, new_call=v)`
  (ie. we update a `lookup_or_generate` to look up a different call, and the new call is `v`)
OR
- `Gen.generate(lookup_or_generate, call=v)`

We also assume that if `update` is run for call `u` which makes a call
`lookup_or_generate` for a call `w`, if the version of the world passed to `u`
has an argdiff which is not `NoChange` for call `w`, an update will be run on
this `lookup_or_generate`, UNLESS this call to `lookup_or_generate` is removed
by the update to `u`. (In which case this update _removes_ a lookup by `u` of `w`.)

We would like to guarantee that once the update algorithm has run:
3a: every new dependency has been added to the dependency graph
3b: every removed dependency has been noted in the dependency graph
3c: the topological order is consistent with the new dependency graph,
ie. if call `u` looks up call `v` in the graph, then `sort[v] < sort[u]`.

#### Proof of 3a:
In the procedure `Gen.generate(lookup_or_generate, v)` OR `Gen.update(lookup_or_generate_trace, v)`,
we account for both of the cases where a new dependency is added.
In this procedure, step 1 is to increment the lookup_counts
which we use to track the dependency graph, so this added
dependency is always tracked.

#### Proof of 3b:
In the "top-level algorithm", step 3 is to scan the discard choicemaps
from all the world updates and the kernel update, and
decrease the lookup counts accordingly.  A fundamental assumption
of the `UsingWorld` combinator is that the discards from the kernel
and memoized generative functions are consistent with the calls
these make. (This assumption is articulated
in the formalism for `UsingWorld`, and error checks are provided
where possible to ensure that this is occurring.)
Therefore any dependencies which are dropped, ie. lookups which
no longer occur, will result in a discard choicemap
which indicates that this lookup used to occur, but no longer does.
Therefore this step 3 will remove in the dependency
graph any dependency which was removed during the update.

#### Proof of 3c:
The proof for this part relies on some invariants which are maintained
during the execution of the algorithm.
The first are:

##### Invariants:

(A) At any time, `fringe_bottom` is such that if call `w` satisfies
`sort[w] < fringe_bottom`, then the algorithm will not call
`update_or_generate(w)` from this point on.

(B) At any time `fringe_top` is such that if call `w` satisfies
`sort[w] > fringe_top`, then call `w` has not yet been visited
by `update_or_generate`.

(C) `fringe_bottom` and `fringe_top` only every increase in value
throughout execution of the algorithm.

Property (C) is evident.  Property (B) is maintained since
whenever `update_or_generate(u)` is called, line 3 increments
`fringe_top` to be at least as large as `sort[u]`.

Property (A) is proven as follows.  We only ever call
`update_or_generate(w)` at line 6.2.2 of the
world update algorithm or in `Gen.update/generate(lookup_or_generate)`.
In `Gen.update/generate`, we only run `update_or_generate(w)` if
`sort[w] >= fringe_bottom`.
In the world update algorithm, we only ever run `update_or_generate(w)`
on a `w` which is the most recent `w` pulled off of `Q`, and thus
the call with the highest index we have pulled off of `Q`.  Since
we only ever update `fringe_bottom` at line 6.2.1.3 of the world update
algorithm by setting `fringe_bottom = sort[u]` for a call `u` we just
pulled off of `Q`, if the world update algorithm calls
`update_or_generate(w)`, the current value of `fringe_bottom` is
either `sort[w]` or `sort[u]` for some `u` pulled off of `Q` before `w`
(or is 0, which we initialize `fringe_bottom` to).  In any case,
`fringe_bottom <= sort[w]`.  Thus we only run `update_or_generate`
on calls with position currently greater than `fringe_bottom`.

Invariants (A), (B), and (C) help us understand 3 other important
invariants:

1. If when we call `update_or_generate(u)`, `sort[v] < fringe_bottom`,
   then after the update, we will have `sort[v] < sort[u]`.

2. at any time, all the vertices in `call_update_order`
   and all the indices in `updated_sort_indices` are between
   `fringe_bottom` and `fringe_top`.

3. If `v` is added to `call_update_order` before `u` is (and both are
   placed on `call_update_order` at some point), we will end the update
   with `sort[v] < sort[u]`

To prove these, we must consider the behavior of the only
procedure which changes topological positions: `reorder_update_cycle!`.  
This procedure only changes positions
for calls in `call_update_order`, and only to positions in
`updated_sort_indices`; furthermore it always empties both of
these data structures when it is called.
(It runs until emptying `call_update_order`,
and in every call to `update_or_generate`, we add exactly
one element to each of these, so they always contain the same
number of elements.  Note that `update_or_generate` is
the only place we add elements to either of these data structures.)

By invariant (B), we have never called `update_or_generate(w)` for
a `w` after `fringe_top`.  Therefore `call_update_order`
`updated_sort_indices` both contain elements before `fringe_top`.
We only call `reorder_update_cycle!`
just before advancing `fringe_top` and `fringe_bottom` to the same
point, past the previous `fringe_top` (or
just before ending the entire update algorithm, in which case
future behavior is irrelevant.)
Therefore by invariant (A), at the end of a call to `reorder_update_cycle!`,
if we are not done with the update algorithm, since we advance
`fringe_bottom` past the previous `fringe_top`, and thus past
all of the elements which were in `call_update_order` before
the call to `reorder_update_cycle!`.  By invariant (A),
we will never call `update_or_generate` on any of these
calls again since they are now before `fringe_bottom`;
thus these elements can never again end up on `call_update_order`
and have their position changed.  Thus invariant (1) holds.
Likewise, since every element we call `update_or_generate` on must
be after `fringe_bottom`, and by the above discussion,
before `fringe_top`, invariant (2) holds.

We also note that if `v` is added to `call_update_order` before `u`,
then `reorder_update_cycle!` will pull `v` off of this queue before `u`
and thus assign `v` a lower index than `u` (since `updated_sort_indices`
is a priority queue and has lower elements pulled off first),
and therefore we will have `sort[v] < sort[u]`, confirming
invariant (3).

##### Proof of 3c:
Now, let's consider an update and prove that
the order we end up with is correct.
Say in the new dependency structure implied by an update,
call `u` looks up call `v`.

There are 2 cases:
i. before this update, call `u` did look up `v`.
ii. before this update, call `u` did not look up `v`.

Case i. before this update, call `u` did look up `v`. 
Note that in this case, before the update, `sort[v] < sort[u]`.

If neither `u` nor `v` are visited by `update_or_generate`
during this update, neither position
will be added to `call_update_order`, so neither will have their position
changed; thus we end with `sort[v] < sort[u]`.

If `u` was visited by `update_or_generate` (regardless of whether
`v` was), then:
- if when we call `update_or_generate(u)`, `fringe_bottom > sort[v]`,
  then by invariant 1 above we know that we end the algorithm
  with `sort[v] < sort[u]`.
- else, when we update `u`, `fringe_bottom <= sort[v] < sort[u]`.
  Then when updating `u`, the diff for the world will reflect that
  there may be a change for `v`, and thus trigger a 
   call to `update(lookup_or_generate_trace, v)`, which will call
   `update_or_generate(v)` and put `v` onto the `call_update_order`
   (or, if `v` was already visited, it has already been put
   onto `call_update_order`).  Only
   then can `u` finish its update and be placed onto `call_update_order`,
   so `v` will be sorted into a position before `u`.

The only other case is that `v` was updated but `u` was not.
Once the update for `v` is complete in `update_or_generate`, if `u < fringe_top`,
`u` is added to `Q` since it depends on `v`,
and thus `u` does get updated, contradicting our assumption.
Thus this case only occurs if at the time the update for `v` is completed,
`sort[v] <= fringe_top < sort[u]`.
By invariant 2, all the calls in `call_update_order` have index before
`fringe_top` and thus before `sort[u]`, and thus `v` is added
to `call_update_order` before any call with position after `sort[u]`;
thus by invariant 3, we end with `sort[v] < sort[u]`.

Case ii. Before this update, `u` did not depend on `v`.
Thus there must have been an update for `u`
where there was a call to `generate(lookup_or_generate, v)` or
`update(lookup_or_generate, old_call=w, new_call=v)` made by `u`.

If during the `u` update, `sort[v] < fringe_bottom`,
then by invariant 1, we end up with `sort[v] < sort[u]`.

If instead during the `u` update, `fringe_bottom <= sort[v]`, lookup_or_generate
will call `update_or_generate(v)` (procedure Gen.generate/Gen.update line 3.1).
If `v` has already been visited,
then either `v` is already on the update stack, which means there is a cycle
and this is an invalid update, or an update for `v` has already completed,
in which case  `v` must already be in `call_update_order`,
so we will end up with `v` before `u`.
Else, the `update_or_generate(v)` will run an update for `v` as needed,
then place `v` on `call_update_order`, before finishing the update
for `u` and placing `u` on `call_update_order`.  Thus by invariant 3 we will
again end up with `sort[v] < sort[u]`.