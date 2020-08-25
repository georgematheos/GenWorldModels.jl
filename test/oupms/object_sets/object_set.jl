@testset "object sets" begin
    # TODO
    println("Reminder: write unit tests for object set types.")
end

@gen (static) function _get_aircraft_set(world, num_aircrafts)
    set ~ GetSingleOriginObjectSet(:Aircraft)(world, (), num_aircrafts)
    return set
end
@load_generated_functions()
get_aircraft_set = UsingWorld(_get_aircraft_set)

@testset "get single origin object set" begin
    tr, weight = generate(get_aircraft_set, (5,))
    @test weight == 0.
    set = get_retval(tr)
    @test set isa ObjectSet
    @test set isa GenWorldModels.SimpleObjectSet
    @test length(set) == 5
    @test all(obj isa AbstractOUPMObject{:Aircraft} for obj in set)
    obj = GenWorldModels._uniform_sample(set)
    @test obj in set

    new_tr, weight, retdiff, discard = update(tr, (6,), (UnknownChange(),), EmptyChoiceMap())
    @test weight == 0.
    @test length(get_retval(new_tr)) == 6
    @test all((obj in get_retval(new_tr)) for obj in get_retval(tr))
    @test !all((obj in get_retval(tr)) for obj in get_retval(new_tr))
end

@dist num_blips(_, _::Tuple{<:OUPMObject{:Aircraft}}) = poisson(2)
get_blip_set = GetOriginIteratingObjectSet(:Blip, (:num_blips,))
@gen (static) function _generate_aircraft_and_blip_set(world, num_aircrafts)
    aircraft_set ~ GetSingleOriginObjectSet(:Aircraft)(world, (), num_aircrafts)
    blip_set ~ get_blip_set(world, ((aircraft_set,),))
    return (aircraft_set, blip_set)
end
@load_generated_functions()
generate_aircraft_and_blip_set = UsingWorld(_generate_aircraft_and_blip_set, :num_blips => num_blips)

@testset "get origin iterating object set" begin
    tr, weight = generate(generate_aircraft_and_blip_set, (3,), choicemap(
        (:world => :num_blips => (Aircraft(1),), 2),
        (:world => :num_blips => (Aircraft(2),), 1),
        (:world => :num_blips => (Aircraft(3),), 2)
    ))
    @test isapprox(weight, sum(logpdf(poisson, v, 2) for v in (2, 1, 2)))
    (aset, bset) = get_retval(tr)
    @test aset isa ObjectSet && bset isa ObjectSet
    @test aset isa GenWorldModels.SimpleObjectSet
    @test bset isa GenWorldModels.EnumeratedOriginObjectSet
    @test all(obj isa AbstractOUPMObject{:Aircraft} for obj in aset)
    @test all(obj isa AbstractOUPMObject{:Blip} for obj in bset)
    display(aset)
    display(bset)
    display(tr.world.id_table)

    @test length(aset) == 3
    @test length(bset) == 5

    new_tr, weight, retdiff, discard = update(tr, (3,), (NoChange(),), choicemap((:world => :num_blips => (Aircraft(2),), 3)))
    @test isapprox(weight, logpdf(poisson, 3, 2) - logpdf(poisson, 1, 2))
    (new_aset, new_bset) = get_retval(new_tr)
    @test new_aset == aset
    @test length(new_bset) == 7
    @test all(obj in new_bset for obj in bset)
    @test !all(obj in bset for obj in new_bset)
end