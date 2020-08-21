function iterator_product(itrs...)
    collect(Iterators.product(itrs...))
end
function iterator_product(itrs::Vararg{<:Diffed{<:Any, UnknownChange}})
    vals = map(strip_diff, itrs)
    Diffed(iterator_product(vals), UnknownChange())
end

@gen (static) function get_single_origin_set(typename, num_statement_addr, origin, world)
    num ~ lookup_or_generate(world[num_statement_addr][origin])
    set ~ get_single_origin_object_set(typename, world, origin, num)
    return set
end

@gen (static) function get_single_origintype_set(typename, num_statement_addr, originset_tuple, world)
    origins = iterator_product(originset_tuple...)
    len = length(origins)
    sets ~ Map(get_single_origin_set)(fill(typename, len), fill(num_statement_addr, len), origins, fill(world, len))
    pairs = [origin => set for (origin, set) in zip(origins, sets)]
    return EnumeratedOriginObjectSet(PersistentHashMap(pairs))
end

@gen (static) function get_origin_iterating_object_set(typename, num_statement_addresses, world, originset_tuples)
    sets ~ Map(get_single_origintype_set)(
        fill(typename, length(num_statement_addresses)),
        num_statement_addresses,
        originset_tuples,
        fill(world, length(num_statement_addresses))
    )
    return length(sets) > 1 ? merge(sets...) : sets[1]
end
@load_generated_functions()

struct GetOriginIteratingObjectSetTrace <: Gen.Trace
    gen_fn
    tr::Gen.get_trace_type(get_origin_iterating_object_set)
end
Gen.get_score(tr::GetOriginIteratingObjectSetTrace) = get_score(tr.tr)
Gen.get_gen_fn(tr::GetOriginIteratingObjectSetTrace) = tr.gen_fn
Gen.get_retval(tr::GetOriginIteratingObjectSetTrace) = get_retval(tr.tr)
Gen.get_args(tr::GetOriginIteratingObjectSetTrace) = get_args(tr.tr)[3:end]
Gen.get_choices(tr::GetOriginIteratingObjectSetTrace) = get_choices(tr.tr)
Gen.project(::GetOriginIteratingObjectSetTrace, ::Selection) = 0.

#=
eg.
GetOriginIteratingObjectSet(
    event -> :world => :num_noise_detections => (),
    (event, station) -> :world => :num_event_detections => (event, station)
)(world, (events_set,), (events_set, station_set))
=#
struct GetOriginIteratingObjectSet <: Gen.GenerativeFunction{EnumeratedOriginObjectSet, GetOriginIteratingObjectSetTrace}
    typename::Symbol
    num_statement_addresses::Tuple
end

function Gen.simulate(gen_fn::GetOriginIteratingObjectSet, args::Tuple)
    tr = simulate(get_origin_iterating_object_set,
        (gen_fn.typename, gen_fn.num_statement_addresses, args...)
    )
    return GetOriginIteratingObjectSetTrace(gen_fn, tr)
end
Gen.generate(gen_fn::GetOriginIteratingObjectSet, args::Tuple, ::EmptyChoiceMap) = (simulate(gen_fn, args), 0.)

function Gen.update(tr::GetOriginIteratingObjectSetTrace, args::Tuple, argdiffs::Tuple, spec::UpdateSpec, eca::Selection)
    args = (tr.gen_fn.typename, tr.gen_fn.num_statement_addresses, args...)
    diffs = (NoChange(), NoChange(), argdiffs...)
    new_tr, weight, retdiff, discard = update(tr.tr, args, argdiffs, spec, eca)
    (GetOriginIteratingObjectSetTrace(tr.gen_fn, new_tr), weight, retdiff, discard)
end