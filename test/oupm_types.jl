@type Blip

@testset "oupm type construction" begin
    @test Blip(1) isa OUPMObject{:Blip}
    @test Blip(1) isa ConcreteIndexOUPMObject{:Blip}
    @test_throws MethodError Blip("i am a string")
    @test Blip((Blip(1), Blip(2)), 4) isa ConcreteIndexOUPMObject{:Blip}
end