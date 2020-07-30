@type Timestep
@type Blip

@testset "oupm type construction" begin
    @test Blip(1) isa OUPMObject{:Blip}
    @test Blip(1) isa ConcreteIndexOUPMObject{:Blip}
    @test_throws MethodError Blip("i am a string")
    @test Blip((Blip(1), Blip(2)), 4) isa ConcreteIndexOUPMObject{:Blip}
    @test ConcreteIndexOUPMObject{:Blip} <: Blip
    @test AbstractOUPMObject{:Blip} <: Blip

    t1 = AbstractOUPMObject{:Timestep}(gensym())
    t2 = AbstractOUPMObject{:Timestep}(gensym())
    @test typeof((Blip((t1, t2), 3).origin)) <: Tuple{Vararg{<:AbstractOUPMObject}}
    @test Blip((t1, t2), 3) isa ConcreteIndexAbstractOriginOUPMObject{:Blip}
    @test !(Blip((t1, Timestep(2)), 3) isa ConcreteIndexAbstractOriginOUPMObject)
    @test Blip((t1, Timestep(2)), 3) isa ConcreteIndexOUPMObject
end