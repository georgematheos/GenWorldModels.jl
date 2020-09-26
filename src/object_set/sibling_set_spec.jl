DIFF_MAY_CAUSE_CHANGE_ERROR_MSG(s) = """
Sibling set sets can only be updated efficiently for num statements which cannot change without an update spec,
and the gen fn at $(s.num_address) cannot be statically confirmed to satisfy this property.
One way to ensure this property is satisfied is to use a `Gen.Distribution` as the number statement generative function.
"""

struct WorldOrOriginChange <: Gen.Diff end
struct WorldChange <: Gen.Diff end

struct SiblingSetSpec
    typename::Symbol
    num_address::CallAddr
    world::World
    origin::Tuple{Vararg{<:OUPMObject}}
end
function Base.getproperty(sss::Diffed{SiblingSetSpec, WorldOrOriginChange}, sym::Symbol)
    stripped = strip_diff(sss)
    if sym === :typename
        Diffed(stripped.typename, NoChange())
    elseif sym === :num_address
        Diffed(stripped.num_address, NoChange())
    elseif sym === :world
        Diffed(stripped.world, WorldUpdateDiff())
    elseif sym === origin
        Diffed(stripped.origin, UnknownChange())
    end
end
function Base.getproperty(sss::Diffed{SiblingSetSpec, WorldChange}, sym::Symbol)
    stripped = strip_diff(sss)
    if sym === :world
        Diffed(stripped.world, WorldUpdateDiff())
    else
        Diffed(getfield(stripped, sym), NoChange())
    end
end

struct GetSiblingSetSpec
    typename::Symbol
    num_address::CallAddr
end
GetSiblingSetSpec(tn::Diffed{Symbol, NoChange}, na::Diffed{<:CallAddr, NoChange}) = GetSiblingSetSpec(strip_diff(tn), strip_diff(na))
function (s::GetSiblingSetSpec)(world::World, origin::Tuple)
    SiblingSetSpec(s.typename, s.num_address, world, origin)
end
function (s::GetSiblingSetSpec)(world::Diffed{<:World, WorldUpdateDiff}, origin::Diffed{<:Tuple})
    # TODO
    # for diff tracking, check whether the num has changed, and whether anything has changed
    # for the origin
    Diffed(s(strip_diff(world), strip_diff(origin)), WorldOrOriginChange())
end
function (s::GetSiblingSetSpec)(world::Diffed{<:World, WorldUpdateDiff}, origin::Diffed{<:Tuple, NoChange})
    # TODO
    # for diff tracking, check whether the num has changed, and whether anything has changed
    # for the origin
    Diffed(s(strip_diff(world), strip_diff(origin)), WorldChange())
end
(s::GetSiblingSetSpec)(world::Diffed{<:World}, origin::Tuple) = s(world, Diffed(origin, NoChange()))

struct GetOriginsToSiblingSetSpecs
    typename::Symbol
    num_address::CallAddr
end
GetOriginsToSiblingSetSpecs(tn::Diffed{Symbol, NoChange}, na::Diffed{<:CallAddr, NoChange}) = GetOriginsToSiblingSetSpecs(strip_diff(tn), strip_diff(na))

function (s::GetOriginsToSiblingSetSpecs)(world::World, origins::AbstractSet)
    # we do the check here instead of during updates since we have to call this
    # to ever perform an update, but we don't want to waste time on the check on every update
    @assert(cannot_change_retval_due_to_diffs(world, s.num_address, typeof(first(origins))), DIFF_MAY_CAUSE_CHANGE_ERROR_MSG(s))
    # TODO: we should use the type from the origins set rather than the type of just one element...
    # unfortunately, types are not being tracked well right now, so this doesn't currently work

    get_spec(origin) = SiblingSetSpec(s.typename, s.num_address, world, origin)
    lazy_set_to_dict_map(get_spec, origins)
end
function (s::GetOriginsToSiblingSetSpecs)(a::Diffed, b::Diffed)
    error("Not implemented")
end
function (s::GetOriginsToSiblingSetSpecs)(world::Diffed{<:World, WorldUpdateDiff}, origins::Diffed{<:PersistentSet, <:Union{NoChange, SetDiff}})
    origins, origins_diff = strip_diff(origins), get_diff(origins)
    in_removed = origins_diff isa SetDiff ? origins_diff.deleted : Set()
    in_added = origins_diff isa SetDiff ? origins_diff.added : Set()

    world = strip_diff(world)
    old_world = _previous_world(world)

    ssspec = GetSiblingSetSpec(s.typename, s.num_address)
    out_removed = Set(in_removed)
    out_added = Dict(origin => ssspec(world, origin) for origin in in_added)
    out_updated = Dict{Any, Diff}()

    updated_num_statement_keys = (key for (key, subtree) in get_subtrees_shallow(get_subtree(world.state.spec, s.num_address)) if !isempty(subtree))
    for origin in Iterators.flatten((
        updated_num_statement_keys,
        _get_origins_with_id_table_updates(world, s.typename)
    ))
        if origin in origins && !(origin in in_added)
            # if we looked this up last time and are still looking it up, it is updated
            out_updated[origin] = WorldChange()
        end
    end

    get_spec(origin) = SiblingSetSpec(s.typename, s.num_address, world, origin)
    Diffed(lazy_set_to_dict_map(get_spec, origins), DictDiff(out_added, out_removed, out_updated))
end