@gen (static, diffs) function a(world, idx)
    b_val ~ lookup_or_generate(world[:b][idx])
    val ~ normal(b_val, 0.05)
    return val
end
@gen function b(world, idx)
    if idx == 1
        val ~ normal(0, 1)
    else
        a_val ~ lookup_or_generate(world[:a][idx - 1])
        val ~ normal(a_val, 1)
    end
    return val
end
@gen (static, diffs) function ab_kernel(world)
    idx ~ uniform_discrete(1, 4)
    a_val ~ lookup_or_generate(world[:a][idx])
    b_val ~ lookup_or_generate(world[:b][idx])
    sum = a_val + b_val
    return sum
end
ab = UsingWorld(ab_kernel, :a => a, :b => b)
@load_generated_functions()

@testset "multiple memoized generative functions" begin
    tr, weight = generate(ab, (), choicemap((:kernel => :idx, 4)))
    @test weight ≈ log(1/4)

    new_tr, weight, retdiff, discard = update(tr, (), (), choicemap(
        (:kernel => :idx, 3),
        (:world => :a => 1 => :val, 0.),
        (:world => :b => 2 => :val, 1.)
    ))

    @test new_tr[:world => :a => 1] == 0.
    @test new_tr[:world => :b => 2] == 1.
    @test new_tr[:world => :b => 1] == tr[:world => :b => 1]
    @test new_tr[:world => :a => 3] == tr[:world => :a => 3]

    new_choices = get_choices(new_tr)
    @test isempty(get_submap(new_choices, :world => :a => :4))
    @test isempty(get_submap(new_choices, :world => :b => :4))

    # a1, b2 changed. a2 depends on b2 so its weight must change. a4, b4 deleted. other lookups shouldn't change
    expected_weight = (
        logpdf(normal, new_tr[:world => :a => 1], new_tr[:world => :b => 1], 0.05) - logpdf(normal, tr[:world => :a => 1], tr[:world => :b => 1], 0.05) # a1 change
      + logpdf(normal, new_tr[:world => :b => 2], new_tr[:world => :a => 1], 1) - logpdf(normal, tr[:world => :b => 2], tr[:world => :a => 1], 1) # b2 change
      + logpdf(normal, new_tr[:world => :a => 2], new_tr[:world => :b => 2], 0.05) - logpdf(normal, tr[:world => :a => 2], tr[:world => :b => 2], 0.05) # a2 change
      - logpdf(normal, tr[:world => :b => 4], tr[:world => :a => 3], 1) - logpdf(normal, tr[:world => :a => 4], tr[:world => :b => 4], 0.05) # a4 and b4 deleted
    )

    @test weight ≈ expected_weight
    @test weight ≈ get_score(new_tr) - get_score(tr)




end