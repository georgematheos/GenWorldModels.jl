#=
Uses this gen fn from `simple.jl`:

@gen function generate_size(world, idx)
    size ~ normal(idx, 1)
    return size
end

@gen function kernel(world)
    idx1 ~ uniform_discrete(1, 10)
    idx2 ~ uniform_discrete(1, 10)
    
    size1 ~ lookup_or_generate(world[:sizes][idx1])
    size2 ~ lookup_or_generate(world[:sizes][idx2])

    total = size1 + size2
    return total
end

const get_total_size = UsingWorld(kernel, :sizes => generate_size)

=#

@testset "other gfi functions" begin

@testset "simulate" begin
    tr = simulate(get_total_size, ())
    ch = get_choices(tr)
    tr2, weight = generate(get_total_size, (), ch)
    @test weight ≈ get_score(tr2)
    @test get_score(tr2) ≈ get_score(tr)
end

# generate, update are tested in the other test files

# regenerate not currently implemented

@testset "trace lookup methods" begin
    # get_args
    # get_retval
    # get_choices
    # get_score
    # get_gen_fn
    # getindex
end

@testset "project" begin
    tr = simulate(get_total_size, ())

    idx1 = tr[:kernel => :idx1]
    idx2 = tr[:kernel => :idx2]
    
    weight = project(tr, select(:world => :sizes => idx1))
    expected_weight1 = logpdf(normal, tr[:world => :sizes => idx1], idx1, 1)
    expected_weight2 = logpdf(normal, tr[:world => :sizes => idx2], idx2, 1)
    @test weight ≈ expected_weight1

    weight = project(tr, select(:kernel => :idx1))
    @test weight ≈ log(1/10)

    weight = project(tr, select(:kernel))
    @test weight ≈ 2*log(1/10)

    weight = project(tr, select(:world))
    @test weight ≈ expected_weight1 + expected_weight2
end

@testset "propose" begin
    (choices, weight, retval) = propose(get_total_size, ())
    tr, weight2 = generate(get_total_size, (), choices)
    @test weight2 ≈ get_score(tr)
    @test weight ≈ get_score(tr)
    @test get_retval(tr) == retval
    @test choices == get_choices(tr)
end

@testset "assess" begin
    @test_throws Exception assess(get_total_size, (), choicemap((:kernel => :idx1, 2)))
    
    choices = choicemap(
        (:kernel => :idx1, 1),
        (:kernel => :idx2, 2),
        (:world => :sizes => 1 => :size, 1.5),
        (:world => :sizes => 2 => :size, 2.1)
    )

    tr, genweight = generate(get_total_size, (), choices)

    (weight, retval) = assess(get_total_size, (), choices)
    @test weight ≈ genweight
    @test weight ≈ get_score(tr)
    @test retval == get_retval(tr)
end

# TODO: gradients

end