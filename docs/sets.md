# Sets in OUPMs

In OUPMs it is frequently useful to construct a set of objects.
These sets can be sampled from, or have a property looked up for
each object in it.

This "mapping a property over an object set" is for instance one way
to produce the return value for some types of world models.

## Outline of my plan

### Extend Gen to support `SetDiff`s, a `SetMap` combinator, and an `NoCollisionSetMap` combinator

`NoCollisionSetMap` is like `SetMap`, but it returns a `PersistentSet` rather than a `MultiSet` since
it assumes the generative function we are mapping through will return a distinct element for every
input.

### Add a `tracked_union` function for merging sets to create a `TrackedUnionSet`

A `TrackedUnionSet` will be aware of what sets it is a union of.  When any of the
sets it is a union of get a `SetDiff`, it will therefore be able to perform a more efficient update.

In the initial version, I will have the `TrackedUnionSet` both store things a set
of all objects, and all the unioned sets.  In future versions, it would make sense to provide
a function mapping from the input object to which input set the object would be a part of
if it were in the set at all.  This would remove the need to have the complete set
to get efficient lookups--and would thus speed up removing parts of the set.

Whether we use a complete set in the underlying representation, or require a function from object
to the single input set it must have come from, we can easily guarantee no duplicates appear in the
union set.  (In the case where we store a complete set, we can just ignore duplicates.  In the case
where we have a function to the input set, if this function is valid, it means each input
could only possibly be in a single set.)

`tracked_union` will be a deterministic generative function so that we can update it
properly.  (If it's a normal function, while it gets the diffs, it doesn't get the old
version of the object set, so we can't update it quite the right way.)

### Add `GenWorldModels` syntactic sugar to handle num lookups & conversions

### Example: Information Extraction

```julia
@gen (static, diffs) function get_facts_for_rel(rel, world)
    ent_pairs ~ get_entity_pairs(world, rel) # returns a set
    origins = no_collision_set_map(entpair -> (rel, entpair...), ent_pairs)
    sib_specs = GetSiblingSpecs(:Fact, :num_facts)(world, origins)
    facts ~ NoCollisionSetMap(get_singleton_object_set)(sib_specs)

    return facts
end

@gen (static, diffs) function construct_obj_set(world, _)
    num_relations ~ @w num_relations[()]
    rel_set ~ get_single_origin_object_set(SiblingSpec(:Relation, :num_relations)(world, ()))
    facts_sets ~ NoCollisionSetMap(get_facts_for_rel)(rel_set, world)
    facts ~ tracked_union(facts_sets)
    return facts
end
```

### Example: Seismic monitoring

```julia
@gen (static, diffs) function get_detections(world, _)
    events ~ GetSingleOriginObjectSet(:Event, :num_events)(world, ())
    stations ~ GetSingleOriginObjectSet(:Station)(world, (), NUM_STATIONS)

    noise_detection_sets ~ GetOriginIteratingObjectSet(:Detection, :num_noise_detections)(world, stations)
    real_detection_sets ~ GetOriginIteratingObjectSet(:Detection, :num_real_detections)(world, events, stations)

    detections = tracked_union(noise_detections, real_detections)

    return detections
end

# where we have something like
@gen function get_origin_iterating_object_set(typename, num_address, world, sets...)
    all_origins = tracked_product_set(sets)
    sibling_specs = GetSiblingSpecs(typename, num_address)(world, origins)
    object_sets ~ NoCollisionSetMap(get_single_origin_object_set)(sibling_specs)
    return tracked_union(object_sets)
end
```

The idea here is that we have a function `get_origins_in_context` which looks at the diffs tracked in
the world to understand
1. which origins have had the number statement changed
2. which origins have a change in the ID table

### Sibling set specs (``contextualized origin'')

For this we need an idea of an ``origin context'', aka a "sibling set specification",
which specifies a set of objects sharing a single origin (hence, a set of "siblings").
A context is effectively a spec for the function `get_single_origin_object_set`,
and specifies:
- An origin object tuple
- The type of object this is an origin for
- The num address for the object type under this origin
- The world these objects exist in

Then `get_single_origin_object_set` constructs the object
set specified by this.  The reason to have this "intermediary step"
is that it gives us a place to propagate argdiffs.  By having a special
function which incrementally updates the set of ``origin context''s for
a collection of origins, we can achieve better asymptotic performance.

## TODOs
- `tracked_union`
- `get_origin_iterating_object_set`
- `tracked_product_set`

Maybe:
- `set_map`

### Completed TODOs
- `SiblingSetSpec`, `GetSiblingSpecs`
  - Make sure we can implement this with proper diff tracking
- Implement `no_collision_set_map` with graceful diff handling
- `get_sibling_set`
- Implement `SetMap`, `NoCollisionSetMap`
