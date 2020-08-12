struct GetObjectSetTrace <: Gen.Trace
    world::World
    set::ObjectSet
    originset_subtraces
end
Gen.get_score(::GetObjectSetTrace) = 0.
Gen.get_gen_fn(tr::GetObjectSetTrace) = get_object_set
Gen.get_retval(tr::GetObjectSetTrace) = tr.set
Gen.get_args(tr::GetObjectSetTrace) = (tr.world,)

# Since I use `generate` calls to track increasing counts, and `discard`s to track decreasing counts,
# so long as I return the correct `discard`, counting is done correctly.
# However, this is a bit of a hack, so I'll leave this as a
# TODO
Gen.get_choices(tr::GetFactGetObjectSetTraceSetTrace) = EmptyChoiceMap()

struct GetObjectSet <: Gen.GenerativeFunction end
get_object_set = GetObjectSet()

function Gen.simulate(gen_fn::GetObjectSet, (world, tname))
    # assuming num statements form a DAG
    size = 0
    origin_to_objects = Dict()
    originset_subtrs = []
    for (origin_typenames, origin_to_addr) in world.num_statements[gen_fn.typename]
        originset_subtraces = Tuple(simulate(lookup_or_generate, world[_get_objset_addr][typename]) for typename in origin_typenames)
        push!(originset_subtrs, originset_subtraces)
        origin_sets = map(get_retval, originset_subtraces)

        origin_itr = origin_sets == () ? ((),) : Iterators.product(origin_sets)
        for origin in origin_itr
            origin_to_objects[origin] = PersistentSet()
            num = lookup_or_generate(world[origin_to_addr(origin)][origin],)
            for i=1:num
                conc = ConcreteIndexOUPMObject{tname}(origin, i)
                abstract = lookup_or_generate(world[_get_abstract_addr][conc])
                origin_to_objects[origin] = push(origin_to_bjects[origin], abstract)
                size += 1
            end
        end
    end

    set = ObjectSet(size, PersistentHashMap(origin_to_objects))
    GetObjectSetTrace(world, set, originset_subtrs)
end

function Gen.update(tr::GetObjectSetTrace, (world,), (::WorldUpdateDiff,), ::EmptyChoiceMap, ::Selection)
    newset = tr.set
    for (i, (origin_typenames, origin_to_addr)) in enumerate(world.num_statements[gen_fn.typename])
        diffed = Diffed(world, WorldUpdateDiff())[_get_objset_addr][key(get_args(trace)[1])]
        updates = map(trace -> update(trace, (strip_diff(diffed),), (get_diff(diffed),), EmptyAddressTree(), EmptyAddressTree()), tr.originset_subtraces[i])
        new_trs = map(x -> x[1], updates)
        new_sets = map(get_retval, new_trs)
        diffs = map(x -> x[3], updates)
        discards = map(x -> x[4], updates)

        for (i, diff) in enumerate(diffs)
            if diff === NoChange()
                continue
            end
            newset = remove_origins(newset, i, diff, tr.originset_subtraces[i])
            newset = add_origins(newset, i, diff, new_sets)
        end
    end
    for (call, diff) in world.state.diffs
        if is_num_call_for(call, tr.tname)
            origin = key(call)
            new_num = get_val(world, call)
            old_num = get_val(tr.world, call)
            for i=1:new_num
                conc = ConcreteIndexOUPMObject{tr.tname}(origin, i)
                if !has_val(tr.world, Call(_get_abstract_addr, conc))
                    newset = add(newset, world, conc)
                end
            end
            for i=1:old_num
                conc = ConcreteIndexOUPMObject{tr.tname}(origin, i)
                if !has_val(world, Call(_get_abstract_addr, conc))
                    newset = remove(newset, world, conc)
                end
            end
        end
    end

    new_tr = GetObjectSetTrace(world, newset, new_originset_subtraces)
    (new_tr, 0., retdiff, discard)
end