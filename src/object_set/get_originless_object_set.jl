struct GetOriginlessObjectSetTrace <: Gen.Trace
    gen_fn
    world::World
    set::SimpleObjectSet
end
Gen.get_score(::GetOriginlessObjectSetTrace) = 0.
Gen.get_gen_fn(tr::GetOriginlessObjectSetTrace) = tr.gen_fn
Gen.get_retval(tr::GetOriginlessObjectSetTrace) = tr.set
Gen.get_args(tr::GetOriginlessObjectSetTrace) = (tr.world,)
Gen.get_choices(tr::GetOriginlessObjectSetTrace) = nothing # TODO

# GetOriginlessObjectSet(typename, num-address)(world)
struct GetOriginlessObjectSet <: Gen.GenerativeFunction{GetOriginlessObjectSetTrace}
    typename::Symbol
    num_address::Symbol
end

function Gen.simulate(gen_fn::GetOriginlessObjectSet, (world,))
    num = lookup_or_generate(world[gen_fn.num_address.first][gen_fn.num_address.second])
    objects = PersistentSet()
    for i=1:num
        conc = ConcreteIndexOUPMObject{gen_fn.typename}(i)
        objects = assoc(objects, lookup_or_generate(world[_get_abstract_addr][conc]))
    end

    GetOriginlessObjectSetTrace(gen_fn, world, SimpleObjectSet(objects))
end

function Gen.update(tr::GetOriginlessObjectSetTrace, (world,), (::WorldUpdateDiff,), ::EmptyChoiceMap, ::Selection)
    added = []
    removed = []
    
end