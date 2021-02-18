# this test uses the model from `modeling_dsl_integration-seismic.jl`

@testset "commands for kernel DSL" begin
    tr, _ = generate(generate_detections, (2,) choicemap(
        @set_number Event() = 5,
        @set_number Detection(Station(1)) = 2,
        @set_number Detection(Station(2)) = 1,
        @set_number Detection(Station(1), Event(1)) = 1,
        @set magnitude[Event(1)] = 1.5,
        @set reading[Detection(Station(1), Event(1), 1)] = 1.51
    ))

    @test @get_number(tr, Event()) == 5
    @test @get_number(tr, Detection(Station(1))) == 2
    @test @get_number(tr, Detection(Station(1), Event(1))) == 1

    @test @get(tr, magnitude[Event(1)]) == 1
    @test @get(tr, reading[Detection(Station(1), Event(1), 1)]) == 1.51

    @test isempty(get_submap(tr, @obsmodel()))

    @test @abstract(tr, Event(1)) isa AbstractOUPMObject{:Event}
    @test @concrete(tr, @abstract(tr, Event(1))) == Event(1)
    @test @abstract(tr, @concrete(tr, @abstract(tr, Event(1)))) == @abstract(tr, Event(1))
    @test @abstract(tr, Detection(Station(1), 2)) isa AbstractOUPMObject{:Detection}
    @test @concrete(tr, @abstract(tr, Detection(Station(1), 2))) == Detection(Station(1), 2)

    @test @index(tr, Event(1)) == 1
    @test @index(tr, Detection(Station(1), 2)) == 2
    @test @origin(tr, Event(1)) == ()
    @test @origin(tr, Detection(Station(1), 2)) == (Station(1),)

    @test @index(tr, @abstract(tr, Event(1))) == 1
    @test @index(tr, @abstract(tr, Detection(Station(1), 2))) == 2
    @test @origin(tr, @abstract(tr, Event(1))) == ()
    @test @origin(tr, @abstract(tr, Detection(Station(1), 2))) == (Station(1),)

    @testset "object set getting" begin
        @test length(@objects(tr, Event())) == 5
        @test @objects(tr, Event) == @objects(tr, Event())
        @test length(@objects(tr, Detection(Station))) == 3
        @test length(@objects(tr, Detection(Station(1)))) == 2
        @test @objects(tr, Event()) isa AbstractSet
        @test all(o isa GenWorldModels.ConcreteIndexOUPMObject{:Event} for o in @objects(tr, Event))
    end
end