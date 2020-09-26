struct GetFactSetTrace <: Gen.Trace
    world::World
    facts::PersistentSet{AbstractOUPMObject{:Fact}}
    lookup_traces::PersistentSet{Trace}
end
Gen.get_score(::GetFactSetTrace) = 0.
Gen.get_gen_fn(tr::GetFactSetTrace) = get_fact_set
Gen.get_retval(tr::GetFactSetTrace) = tr.facts
Gen.get_args(tr::GetFactSetTrae) = (tr.world,)
function Gen.get_choices(tr::GetFactSetTrace)
    ch = choicemap()
    for (i, tr) in enumerate(tr.lookup_traces)
        set_submap!(ch, i, get_choices(tr))
    end
    ch
end

struct GetFactSet <: Gen.GenerativeFunction end
get_fact_set = GetFactSet()

@gen (static, diffs) function get_originless_nums(world)
    num_entities ~ lookup_or_generate(world[:args][:num_entities])
    num_relations ~ lookup_or_generate(world[:num_relations][()])
    return (num_entities, num_relations)
end
@load_generated_functions()

function Gen.simulate(::GetFactSet, (world,))
    num_e_tr = simulate(lookup_or_generate, world[:args][:num_entities])
    num_r_tr = simulate(lookup_or_generate, world[:num_relations][()])

    facts = Set()
    trs = Set()
    push!(trs, num_e_tr); push!(trs, num_r_tr)
    num_entities, num_relations = get_retval(num_e_tr), get_retval(num_r_tr)
    for e1=1:num_entities, e2=1:num_entities, r=1:num_relations
        tr = simulate(lookup_or_generate, world[:num_facts][(r, e1, e2)])
        push!(trs, tr)
        for idx=1:get_retval(tr)
            fact_tr = simulate(lookup_or_generate, world[:abstract][Fact((r, e1, e2), idx)])
            push!(facts, get_retval(fact_tr))
            push!(trs, fact_tr)
        end
    end

    GetFactSetTrace(world, PersistentSet(facts), PersistentSet(trs))
end

function Gen.update(tr::GetFactSetTrace, (world,), (::WorldUpdateDiff,), ::EmptyChoiceMap, ::Selection)
    # TODO: don't iterate through all the diffs!
    facts = tr.facts
    trs = tr.lookup_traces
    for (call, diff) in world.state.diffs
        if addr(call) == :num_facts
            
        elseif addr(call) == :num_relations
            new_relations = (
                rel for rel in instantiated_abstract_objects(world, Relation)
                    if !is_instantiated(tr.world, rel)
            )
            removed_relations = (
                rel for rel in instantiated_abstract_objects(tr.world, Relation)
                    if !is_instantiated(world, rel)
            )
            for rel in new_relations
                
            end
            for rel in removed_relations
                for e1=
                trs = dissoc(trs, )
            end
        end
        # We assume `num_entities` does not change
    end
end