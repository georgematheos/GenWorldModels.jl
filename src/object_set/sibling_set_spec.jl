DIFF_MAY_CAUSE_CHANGE_ERROR_MSG(s) = """
Sibling set sets can only be updated efficiently for num statements which cannot change without an update spec,
and the gen fn at $(s.num_address) cannot be statically confirmed to satisfy this property.
One way to ensure this property is satisfied is to use a `Gen.Distribution` as the number statement generative function.
"""

struct SiblingSetSpec
    typename::Symbol
    num_address::CallAddr
    world::World
    origin::Tuple{Vararg{<:OUPMObject}}
end

struct GetSiblingSetSpec
    typename::Symbol
    num_address::CallAddr
end
function (s::GetSiblingSetSpec)(world::World, origin::Tuple)
    SiblingSetSpec(s.typename, s.num_address, world, origin)
end
function (s::GetSiblingSetSpec)(world::Diffed{World}, origin::Diffed{<:Tuple})
    # TODO
    # for diff tracking, check whether the num has changed, and whether anything has changed
    # for the origin
end

struct GetSiblingSetSpecs
    typename::Symbol
    num_address::CallAddr
end

struct LazySiblingSetSpecSet <: AbstractSet{SiblingSetSpec}
    s::GetSiblingSetSpec
    world::World
    origins::AbstractSet{<:Tuple}
end
function LazySiblingSetSpecSet(s::GetSiblingSetSpecs, w::World, o::AbstractSet{<:Tuple})
    LazySiblingSetSpecSet(GetSiblingSetSpec(s.typename, s.num_address), w, o)
end
function Base.in(spec::SiblingSetSpec, set::LazySiblingSetSpecSet)
    (
        spec.typename == set.s.typename &&
        spec.num_address == set.s.num_address &&
        spec.world == set.world &&
        spec.origin in set.origins
    )
end
Base.length(spec::LazySiblingSetSpecSet) = length(spec.origins)
_iterator(set::LazySiblingSetSpecSet) = (set.s(set.world, origin) for origin in set.origins)
Base.iterate(set::LazySiblingSetSpecSet) = iterate(_iterator(set))
Base.iterate(set::LazySiblingSetSpecSet, st) = iterate(_iterator(set), st)

function (s::GetSiblingSetSpecs)(world::World, origins::AbstractSet{<:Tuple})
    # we do the check here instead of during updates since we have to call this
    # to ever perform an update, but we don't want to waste time on the check on every update
    @assert(cannot_change_retval_due_to_diffs(world, s.num_address), DIFF_MAY_CAUSE_CHANGE_ERROR_MSG(s))

    LazySiblingSetSpecSet(s, world, origins)
end
(s::GetSiblingSetSpecs)(::Diffed, ::Diffed) = error("Not implemented")
function (s::GetSiblingSetSpecs)(world::Diffed{<:World, WorldUpdateDiff}, origins::Diffed{<:PersistentSet, <:Union{NoChange, SetDiff}})
    origins, origins_diff = strip_diff(origins), get_diff(origins)
    in_removed = origins_diff isa SetDiff ? origins_diff.deleted : Set()
    in_added = origins_diff isa SetDiff ? origins_diff.added : Set()
    world = strip_diff(world)
    old_world = _previous_world(world)

    ssspec = GetSiblingSetSpec(s.typename, s.num_address)
    out_removed = Set(no_collision_set_map(origin -> ssspec(old_world, origin), in_removed))
    out_added = Set(no_collision_set_map(origin -> ssspec(world, origin), in_added))

    updated_num_statement_keys = (key for (key, subtree) in get_subtrees_shallow(get_subtree(world.state.spec, s.num_address)) if !isempty(subtree))
    for origin in Iterators.flatten((
        updated_num_statement_keys,
        _get_origins_with_id_table_updates(world, s.typename)
    ))
        if origin in origins
            push!(out_added, ssspec(world, origin))

            # if we looked this up last time, remove the old call
            if !(origin in in_added)
                push!(out_removed, ssspec(old_world, origin))
            end
        end
    end

    Diffed(LazySiblingSetSpecSet(s, world, origins), SetDiff(out_added, out_removed))
end