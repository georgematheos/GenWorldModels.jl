# get_aircraft_origins(n) = [(Aircraft(i),) for i=1:n]
# get_aircraft_origins(n::Diffed) = Diffed(get_aircrafts(strip_diff(n)), UnknownChange())
@dist num_aircrafts(_, _) = poisson(3)
@dist num_blips(_, _) = poisson(2)
@gen (static, diffs) function _get_blip_sib_specs(world)
    num_aircrafts ~ lookup_or_generate(world[:num_aircrafts][()])
    aircrafts ~ GetSingleOriginObjectSet(:Aircraft)(world, (), num_aircrafts)
    origins ~ no_collision_set_map(tuple, aircrafts)
    sibspecs = ((GenWorldModels.GetSiblingSetSpecs)(:Blip, :num_blips))(world, origins)

    nums ~ Map(lookup_or_generate)(mgfcall_map(world[:num_blips], origins))

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
    
    println("deleted:")
    display(retdiff.deleted)
    println("added:")
    display(retdiff.added)

    @test length(retdiff.deleted) == 1 && olds[3] in retdiff.deleted
    @test length(retdiff.added) == 2 && new1 in retdiff.added && new2 in retdiff.added

    @test all(
        GenWorldModels.SiblingSetSpec(:Blip, :num_blips, tr.world, (ac,)) in get_retval(tr)
        for ac in (new1, new2, olds[1], olds[2])
    )
end