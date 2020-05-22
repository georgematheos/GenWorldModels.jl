@gen (static, diffs) function get_num_indices(world, arg::Tuple{})
    num ~ poisson(5)
    return num
end

@gen function add_noise_to_random_dependency(world, n)
    num_indices ~ lookup_or_generate(world[:num_indices][()])
    if n == 1
        val ~ normal(0, 1)
    else
        idx ~ uniform_discrete(1, num_indices)
        old_val ~ lookup_or_generate(world[:vals][idx])
        val ~ normal(old_val, 1)
    end
    return val
end

@gen (static, diffs) function random_noise_dep_kernel(world)
    num_indices ~ lookup_or_generate(world[:num_indices][()])
    val_to_lookup ~ uniform_discrete(1, num_indices)
    last_val ~ lookup_or_generate(world[:vals][val_to_lookup])
    return last_val
end

generate_random_noise_dependencies = UsingWorld(random_noise_dep_kernel, :num_indices => get_num_indices, :vals => add_noise_to_random_dependency)
@load_generated_functions()

@testset "simple dependency updates" begin
    # initially, dependency structure is 4 looks up 3 looks up 2 looks up 1
    tr, weight = generate(generate_random_noise_dependencies, (), choicemap(
        (:world => :num_indices => () => :num, 4),
        (:kernel => :val_to_lookup, 4),
        (:world => :vals => 2 => :idx, 1),
        (:world => :vals => 3 => :idx, 2),
        (:world => :vals => 4 => :idx, 3)
    ))
    logprob_of_constrained_choices = logpdf(poisson, 4, 5) + 4*log(1/4) # constrained num_indices + 4 uniform_discretes
    @test weight ≈ logprob_of_constrained_choices

    ### dependency update causing call drop ###

    # update the dependency structure to 2 looks up 4 looks up 3 looks up 1
    # however, do not change what value the kernel looks up (4)
    # this means that now 2 is irrelevant to the kernel, so the call for 2 should be dropped
    # this means that the fact we provided a constraint for 2 should throw an error
    @test_throws Exception update(tr, (), (), choicemap(
        (:world => :vals => 3 => :idx, 1),
        (:world => :vals => 2 => :idx, 4)
    ))
    new_tr, weight, retdiff, discard = update(tr, (), (), choicemap(
        (:world => :vals => 3 => :idx, 1)
    ))

    # should have dropped call 2
    @test !isempty(get_submap(discard, :world => :vals => 2))
    @test isempty(get_submap(get_choices(new_tr), :world => :vals => 2))
    
    # although the dependency structure should have changed, none of the retvals should have changed
    @test all(new_tr[:world => :vals => i] == tr[:world => :vals => i] for i in [1, 3, 4])
    
    expected_score_of_vals2 = log(1/4) + logpdf(normal, tr[:world => :vals => 2], tr[:world => :vals => 1], 1)
    expected_weight_for_vals3 = logpdf(normal, new_tr[:world => :vals => 3], new_tr[:world => :vals => 1], 1) - logpdf(normal, tr[:world => :vals => 3], tr[:world => :vals => 2], 1)
    expected_weight = expected_weight_for_vals3 - expected_score_of_vals2
    @test weight ≈ expected_weight
    @test get_score(new_tr) - get_score(tr) ≈ weight

    ### dependency update without call drop ###
    # change to 2 looks up 4 looks up 3 looks up 1; also change kernel to lookup 2
    new_tr, weight, retdiff, discard = update(tr, (), (), choicemap(
        (:world => :vals => 3 => :idx, 1),
        (:world => :vals => 2 => :idx, 4),
        (:kernel => :val_to_lookup, 2)
    ))
    
    # although the dependency structure should have changed, none of the retvals should have changed
    @test all(new_tr[:world => :vals => i] == tr[:world => :vals => i] for i in [1, 2, 3, 4])
    expected_weight_for_vals3 = logpdf(normal, new_tr[:world => :vals => 3], new_tr[:world => :vals => 1], 1) - logpdf(normal, tr[:world => :vals => 3], tr[:world => :vals => 2], 1)
    expected_weight_for_vals2 = logpdf(normal, new_tr[:world => :vals => 2], new_tr[:world => :vals => 4], 1) - logpdf(normal, tr[:world => :vals => 2], tr[:world => :vals => 1], 1)
    @test weight ≈ expected_weight_for_vals3 + expected_weight_for_vals2
    @test get_score(new_tr) - get_score(tr) ≈ weight

    @test has_value(discard, :world => :vals => 2 => :idx)
    @test has_value(discard, :world => :vals => 3 => :idx)
end