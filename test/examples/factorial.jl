@gen function approx_fact_helper(world, n)
    if n == 0
        return {:val} ~ normal(1, 0.05)
    else
        prev = lookup_or_generate(world[:fact][n - 1])
        return {:val} ~ normal(n * prev, 0.05)
    end
end
@gen (static, diffs) function fact_kern(world, n)
    val ~ lookup_or_generate(world[:fact][n])
    return val
end
approx_factorial = UsingWorld(fact_kern, :fact => approx_fact_helper)
@load_generated_functions()

@testset "simple recursion - approx_factorial" begin
    tr, weight = generate(approx_factorial, (4,))
    
    fact_vals = [tr[:world => :fact => i] for i=0:4]
    fact_val_scores = [logpdf(normal, fact_vals[1], 1, 0.05)]
    for i=1:4
        push!(fact_val_scores, logpdf(normal, fact_vals[i + 1], i * fact_vals[i], 0.05))
    end
    @test get_score(tr) ≈ sum(fact_val_scores)
    @test weight == 0. # since totally unconstrained

    new_tr, weight, retdiff, discard = update(tr, (4,), (NoChange(),), choicemap((:world => :fact => 0 => :val, 1.)))

    @test !isempty(get_submap(discard, :world => :fact => 0))

    fact_vals = [new_tr[:world => :fact => i] for i=0:4]
    fact_val_scores = [logpdf(normal, fact_vals[1], 1, 0.05)]
    for i=1:4
        push!(fact_val_scores, logpdf(normal, fact_vals[i + 1], i * fact_vals[i], 0.05))
    end
    @test get_score(new_tr) ≈ sum(fact_val_scores)

    @test weight ≈ get_score(new_tr) - get_score(tr)
end