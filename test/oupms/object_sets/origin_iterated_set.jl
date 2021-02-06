# from previous file:
# @dist num_aircrafts(_, _) = poisson(3)
# @dist num_blips(_, _) = poisson(2)
@dist num_false_alarms(_, _) = poisson(1)
@gen (static, diffs) function _get_all_blips(world, num_timesteps)
    aircrafts ~ get_sibling_set(:Aircraft, :num_aircrafts, world, ())
    timesteps ~ get_sibling_set_from_num(:Timestep, world, (), num_timesteps)
    real_blips ~ get_origin_iterated_set(:Blip, :num_blips, world, constlen_vec(aircrafts, timesteps))
    false_blips ~ get_origin_iterated_set(:Blip, :num_false_alarms, world, constlen_vec(timesteps,))
    blips ~ tracked_union(real_blips, false_blips)
    return blips
end
@load_generated_functions()
get_all_blips = UsingWorld(_get_all_blips,
    :num_aircrafts => num_aircrafts,
    :num_blips => num_blips,
    :num_false_alarms => num_false_alarms
)

@testset "set creation with origin iteration" begin
    tr, _ = generate(get_all_blips, (2,), choicemap(
        (:world => :num_aircrafts => (), 3),
        (:world => :num_blips => (Aircraft(1), Timestep(1)), 1),
        (:world => :num_blips => (Aircraft(1), Timestep(2)), 1),
        (:world => :num_blips => (Aircraft(2), Timestep(1)), 2),
        (:world => :num_blips => (Aircraft(2), Timestep(2)), 2),
        (:world => :num_blips => (Aircraft(3), Timestep(1)), 3),
        (:world => :num_blips => (Aircraft(3), Timestep(2)), 3),
        (:world => :num_false_alarms => (Timestep(1),), 1),
        (:world => :num_false_alarms => (Timestep(2),), 0),
    ))

    @test all(obj isa AbstractOUPMObject{:Blip} for obj in get_retval(tr))
    @test length(get_retval(tr)) == 1+1+2+2+3+3+1

    Blp(a, t, i) = Blip((Aircraft(a), Timestep(t)), i)
    spec = WorldUpdate(
        (
            Merge(Aircraft(1), 1, 2,
            (
                Blp(1, 1, 1) => Blp(1, 1, 1),
                Blp(1, 2, 1) => Blp(1, 2, 1),
                Blp(2, 1, 1) => Blp(1, 1, 3),
                Blp(2, 1, 2) => Blp(1, 1, 2),
                Blp(2, 2, 1) => nothing, # delete this blip
                Blp(2, 2, 2) => Blp(1, 2, 2)
            )),
        ),
        choicemap(
            (:world => :num_aircrafts => (), 2),
            (:world => :num_blips => (Aircraft(1), Timestep(1)), 3),
            (:world => :num_blips => (Aircraft(1), Timestep(2)), 2)
        )
    )
    new_tr, _, retdiff, _ = update(tr, (3,), (NoChange(),), spec, AllSelection())

    @test length(get_retval(new_tr)) == 1+1+2+2+3+3+1 - 1
    should_be_removed = GenWorldModels.convert_to_abstract(tr.world, Blp(2, 2, 1))
    @test !(should_be_removed in get_retval(new_tr))
    @test all(obj in get_retval(tr) for obj in get_retval(new_tr))
    @test all(obj in get_retval(new_tr) for obj in get_retval(tr) if obj != should_be_removed)

    @test retdiff.deleted == Set([should_be_removed])
    @test retdiff.added == Set()

    new_tr, _, retdiff, _ = update(tr, (3,), (UnknownChange(),), choicemap(
        (:world => :num_false_alarms => (Timestep(3),), 3),
        (:world => :num_blips => (Aircraft(1), Timestep(3)), 0),
        (:world => :num_blips => (Aircraft(2), Timestep(3)), 0),
        (:world => :num_blips => (Aircraft(3), Timestep(3)), 0)
    ))
    @test length(retdiff.added) == 3
    @test length(retdiff.deleted) == 0
    @test length(get_retval(new_tr)) == length(get_retval(tr)) + 3
    @test all(obj in get_retval(new_tr) for obj in get_retval(tr))
    @test all(obj in get_retval(tr) for obj in get_retval(new_tr) if !(obj in retdiff.added))
end