@testset "address_filter_choicemap" begin
    choices = choicemap(
        (:a => 1),
        (:b => :c, 1),
        (:c => :d, 5),
        (:d => :e, 1),
        (:d => :c, 1)
    )
    no_c_choices = AddressFilterChoicemap(choices, addr -> addr != :c)

    @test has_value(no_c_choices, :a)
    @test isempty(get_submap(no_c_choices, :b))

    println(get_submap(no_c_choices, :b))

    @test isempty(get_submap(no_c_choices, :c))
    @test length(collect(get_values_shallow(get_submap(no_c_choices, :d)))) == 1
    @test no_c_choices[:a] == 1
    @test_throws KeyError no_c_choices[:b => :c]
    @test_throws KeyError no_c_choices[:c => :d]
    @test no_c_choices[:d => :e] == 1
    @test !isempty(get_submap(no_c_choices, :d))
end