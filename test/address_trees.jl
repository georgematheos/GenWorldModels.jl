@type IDTestType1
@type IDTestType2

@testset "address trees" begin
@testset "address_filter_choicemap" begin
    choices = choicemap(
        (:a => 1),
        (:b => :c, 1),
        (:c => :d, 5),
        (:d => :e, 1),
        (:d => :c, 1)
    )
    no_c_choices = GenWorldModels.AddressFilterChoiceMap(choices, addr -> addr != :c)

    @test has_value(no_c_choices, :a)
    @test isempty(get_submap(no_c_choices, :b))
    submap = get_submap(no_c_choices, :b)

    @test isempty(get_submap(no_c_choices, :c))
    @test length(collect(get_values_shallow(get_submap(no_c_choices, :d)))) == 1
    @test no_c_choices[:a] == 1
    @test_throws Gen.ChoiceMapGetValueError no_c_choices[:b => :c]
    @test_throws Gen.ChoiceMapGetValueError no_c_choices[:c => :d]
    @test no_c_choices[:d => :e] == 1
    @test !isempty(get_submap(no_c_choices, :d))
end

@testset "ConvertKeyAtDepthAddressTree" begin
    cm = choicemap(
        (0.5, 2),
        (2 => 3, 4),
        (3 => 4 => 5, 6)
    )
    squared2nd = GenWorldModels.ConvertKeyAtDepthAddressTree{Value, 2}(cm, x -> x^2, sqrt)
    @test get_value(squared2nd, 0.5) == 2
    @test get_submap(squared2nd, 2) == choicemap((9, 4))
    @test get_submap(squared2nd, 3) == choicemap((16 => 5, 6))
    @test length(collect(get_subtrees_shallow(squared2nd))) == 3
    for (addr, tree) in get_subtrees_shallow(squared2nd)
        if addr == 2 || addr == 3
            @assert tree isa GenWorldModels.ConvertKeyAtDepthAddressTree
        else
            @assert tree isa Value
        end
    end
end

@testset "to_abstract_repr and to_concrete_repr" begin
    world = World((), (), NamedTuple())
    a11 = GenWorldModels.generate_abstract_object!(world, IDTestType1(1))
    a12 = GenWorldModels.generate_abstract_object!(world, IDTestType1(2))
    a21 = GenWorldModels.generate_abstract_object!(world, IDTestType2(1))
    a25 = GenWorldModels.generate_abstract_object!(world, IDTestType2(5))

    id_choicemap = choicemap(
        (:vol => a11 => :val, 11),
        (:vol => a12 => :val, 12),
        (:vol => a21 => :val, 21),
        (:vol => a25 => :val, 25),
        (:num => (a11, a21) => :val, 11 + 21) # tuples should be converted too
    )
    idx_choicemap = choicemap(
        (:vol => IDTestType1(1) => :val, 11),
        (:vol => IDTestType1(2) => :val, 12),
        (:vol => IDTestType2(1) => :val, 21),
        (:vol => IDTestType2(5) => :val, 25),
        (:num => (IDTestType1(1), IDTestType2(1)) => :val, 11 + 21)
    )

    @test GenWorldModels.to_abstract_repr(world, idx_choicemap) == id_choicemap
    @test GenWorldModels.to_concrete_repr(world, id_choicemap) == idx_choicemap

    # we should be able to convert tuples with some abstract and somce concrete
    id_choicemap2 = choicemap(
        (:vol => a11 => :val, 11),
        (:vol => a12 => :val, 12),
        (:vol => a21 => :val, 21),
        (:vol => a25 => :val, 25),
        (:num => (a11, IDTestType2(1)) => :val, 11 + 21) # tuples should be converted too
    )
end

end