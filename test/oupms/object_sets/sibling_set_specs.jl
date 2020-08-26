# get_aircraft_origins(n) = [(Aircraft(i),) for i=1:n]
# get_aircraft_origins(n::Diffed) = Diffed(get_aircrafts(strip_diff(n)), UnknownChange())
@dist num_aircrafts(_, _) = poisson(3)
@dist num_blips(_, _) = poisson(2)
@gen (static, diffs) function _get_blip_sib_specs(world)
    aircrafts ~ get_sibling_set(GenWorldModels.GetSiblingSetSpec(:Aircraft, :num_aircrafts)(world, ()))
    origins ~ no_collision_set_map(tuple, aircrafts)
    sibspecs = ((GenWorldModels.GetSiblingSetSpecs)(:Blip, :num_blips))(world, origins)

    collected_origins = collect(origins)
    nums ~ Map(lookup_or_generate)(mgfcall_map(world[:num_blips], collected_origins))
    # blips ~ Map(GetSingleOriginObjectSet(:Blip))(fill(world, length(origins)), collected_origins, nums)

    return sibspecs
end
@load_generated_functions()
get_blip_sib_specs = UsingWorld(_get_blip_sib_specs,
    :num_aircrafts => num_aircrafts,
    :num_blips => num_blips
)

@testset "sibling set specs" begin
    tr, _ = generate(get_blip_sib_specs, (), choicemap(
        (:world => :num_aircrafts => (), 3),
        (:world => :num_blips => (Aircraft(1),), 1),
        (:world => :num_blips => (Aircraft(2),), 2),
        (:world => :num_blips => (Aircraft(3),), 3)
    ))
    @test get_retval(tr) isa AbstractSet{GenWorldModels.SiblingSetSpec}
    setspecs = collect(get_retval(tr))
    @test length(setspecs) == 3
    origins = map(setspec -> setspec.origin, setspecs)
    @test length(unique(origins)) == 3
    @test all(origin isa Tuple{AbstractOUPMObject{:Aircraft}} for origin in origins)
    @test all(GenWorldModels.SiblingSetSpec(:Blip, :num_blips, tr.world, origin) in get_retval(tr) for origin in origins)

    spec = GenWorldModels.UpdateWithOUPMMovesSpec(
        (SplitMove(Aircraft(3), 1, 2, (
            # Blip((Aircraft(3),), 1) => Blip((Aircraft(1),), 1),
            # Blip((Aircraft(3),), 2) => Blip((Aircraft(2),), 2),
            # Blip((Aircraft(3),), 3) => Blip((Aircraft(2),), 1)
        )),),
        choicemap(
            (:world => :num_aircrafts => (), 4),
            (:world => :num_blips => (Aircraft(1),), 1),
            (:world => :num_blips => (Aircraft(2),), 2),
        )
    )
    new_tr, _, retdiff, _ = update(tr, (), (), spec, AllSelection())
    
    # this update should cause a change for the siblingspec for the old Aircraft(3)
    # and the new Aircraft(1) and Aircraft(2).
    # the old Aircraft(1)&Aircraft(2) (new Aircraft(3)&Aircraft(4)) should have no change
    @test retdiff isa SetDiff
    olds = [GenWorldModels.convert_to_abstract(tr.world, Aircraft(i)) for i=1:3]
    new1 = GenWorldModels.convert_to_abstract(new_tr.world, Aircraft(1))
    new2 = GenWorldModels.convert_to_abstract(new_tr.world, Aircraft(2))

    @test length(retdiff.deleted) == 1 && (olds[3],) === collect(retdiff.deleted)[1].origin
    added_origins = map(x -> x.origin, collect(retdiff.added))
    @test length(retdiff.added) == 2 && (new1,) in added_origins && (new2,) in added_origins

    new_origins = map(x -> x.origin, collect(get_retval(new_tr)))
    @test all(
        (ac,) in new_origins
        for ac in (new1, new2, olds[1], olds[2])
    )
end