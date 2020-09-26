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

@testset "simple generate" begin
    diff_idx_tr, diff_weight = generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 2), (:world => :sizes => 1 => :size, 1.05)))
    same_idx_tr, same_weight = generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 1), (:world => :sizes => 1 => :size, 1.05)))

    @test length(collect(get_submaps_shallow(get_submap(get_choices(diff_idx_tr), :world => :sizes)))) == 2
    @test length(collect(get_submaps_shallow(get_submap(get_choices(same_idx_tr), :world => :sizes)))) == 1
    @test diff_idx_tr[:world => :sizes => 1 => :size] == 1.05
    @test diff_idx_tr[:world => :sizes => 2 => :size] != 1.05 # this should be a 0 probability event
    @test same_idx_tr[:world => :sizes => 1 => :size] == 1.05

    @test get_score(same_idx_tr) ≈ log(1/10) + log(1/10) + logpdf(normal, 1.05, 1, 1)
    @test get_score(diff_idx_tr) ≈ get_score(same_idx_tr) + logpdf(normal, diff_idx_tr[:world => :sizes => 2 => :size], 2, 1)
    @test same_weight ≈ get_score(same_idx_tr) # this should have been fully constrained
    @test diff_weight ≈ same_weight
end

# some versions of this which should cause errors:
@gen function broken_kernel1(world)
    idx1 ~ uniform_discrete(1, 10)
    idx2 ~ uniform_discrete(1, 10)
    
    size1 ~ lookup_or_generate(world[:sizes][idx1])
    lookup_or_generate(world[:sizes][idx2]) # untraced call to `lookup_or_generate` is an ERROR!
    size2 ~ lookup_or_generate(world[:sizes][idx2])

    total = size1 + size2
    return total
end
@gen function broken_kernel2(world)
    idx1 ~ uniform_discrete(1, 10)
    idx2 ~ uniform_discrete(1, 10)
    
    size1 ~ lookup_or_generate(world[:sizes][idx1])
    size2 = lookup_or_generate(world[:sizes][idx2]) # untraced call to `lookup_or_generate` is an ERROR!

    total = size1 + size2
    return total
end
@gen function broken_kernel3(world)
    idx1 ~ uniform_discrete(1, 10)
    idx2 ~ uniform_discrete(1, 10)
    
    size1 ~ lookup_or_generate(world[:sizes][idx1])
    size2 = world[:sizes][idx2] # lookups without `lookup_or_generate` should break things!

    total = size1 + size2
    return total
end

broken_get_total_size1 = UsingWorld(broken_kernel1, :sizes => generate_size)
broken_get_total_size2 = UsingWorld(broken_kernel2, :sizes => generate_size)
broken_get_total_size3 = UsingWorld(broken_kernel3, :sizes => generate_size)

@testset "simple errors" begin
    # should throw error if we try to use address :kernel
    @test_throws Exception UsingWorld(kernel, :kernel => generate_size)
    # we should throw an error if we don't use all the constraints we provide to generate
    # TODO: mabe we should change this behavior
    @test_throws Exception generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 2), (:world => :sizes => 3 => :size, 3.05)))
    
    # test error throwing behavior when we don't trace world accesses properly
    @test_throws Exception generate(broken_get_total_size1, ())
    @test_throws Exception generate(broken_get_total_size2, ())
    @test_throws Exception generate(broken_get_total_size3, ())
end

@gen (static, diffs) function static_gen_size(world, idx)
    size ~ normal(idx, 1)
    return size
end

@gen (static, diffs) function static_kernel(world)
    idx1 ~ uniform_discrete(1, 10)
    idx2 ~ uniform_discrete(1, 10)

    size1 ~ lookup_or_generate(world[:sizes][idx1])
    size2 ~ lookup_or_generate(world[:sizes][idx2])

    total = size1 + size2
    return total

end
const static_get_total_size = UsingWorld(static_kernel, :sizes => static_gen_size)
@load_generated_functions()

function simple_update_tests(get_total_size)
    tr, weight = generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 2)))
    @test weight ≈ 2*log(1/10)

    @testset "update memoized values without changing lookups" begin
        new_tr, weight, retdiff, discard = update(tr, (), (), choicemap((:world => :sizes => 1 => :size, 1.000)))
        @test new_tr[:world => :sizes => 1 => :size] == 1.000
        @test weight ≈ logpdf(normal, 1.000, 1, 1) - logpdf(normal, tr[:world => :sizes => 1 => :size], 1, 1)
        @test get_score(new_tr) - get_score(tr) ≈ weight
        
        expected_discard = choicemap(
            (:world => :sizes => 1 => :size, tr[:world => :sizes => 1 => :size])
        )
        
        @test discard == expected_discard
    end
    
    @testset "remove lookup" begin
        new_tr, weight, retdiff, discard = update(tr, (), (), choicemap((:kernel => :idx2, 1)))

        # there should no longer be a choicemap for :world => :sizes => 2
        @test get_submap(get_choices(new_tr), :world => :sizes => 2) == EmptyChoiceMap()

        # both idx1 and idx2 should now have the same val
        @test new_tr[:kernel => :idx2] == 1
        @test new_tr[:kernel => :idx1] == 1
        @test get_retval(new_tr) ≈ 2 * tr[:world => :sizes => 1]
        
        @test weight ≈ -logpdf(normal, tr[:world => :sizes => 2 => :size], 2, 1)
        @test get_score(new_tr) - get_score(tr) ≈ weight
        
        expected_discard = choicemap(
            (:kernel => :idx2, 2),
            (:world => :sizes => 2 => :size, tr[:world => :sizes => 2 => :size])
        )
        @test discard == expected_discard
    end
    @testset "change lookup" begin
        new_tr, weight, retdiff, discard = update(tr, (), (), choicemap((:kernel => :idx2, 3)))
        
        # there should no longer be a choicemap for :world => :sizes => 2
        @test get_submap(get_choices(new_tr), :world => :sizes => 2) == EmptyChoiceMap()
        
        @test new_tr[:kernel => :idx2] == 3

        # the weight should simply be the negative logpdf of the dropped value
        # (we don't include the new value in the weight since the choices were totally unconstrained)
        @test weight ≈ -logpdf(normal, tr[:world => :sizes => 2 => :size], 2, 1)
        
        # another way to look at it is that since there is no untraced randomness,
        # exp(weight) = P(new_trace) / (P(old_trace) * Q(generated values | constraints))
        # here, q(generated vals | constraints) = q(:world => :sizes => 3 is given the value it was given)
        @test weight ≈ get_score(new_tr) - get_score(tr) - logpdf(normal, new_tr[:world => :sizes => 3 => :size], 3, 1)
            
        expected_discard = choicemap(
            (:kernel => :idx2, 2),
            (:world => :sizes => 2 => :size, tr[:world => :sizes => 2 => :size])
        )

        @test discard == expected_discard
    end
    @testset "add lookup" begin 
        tr, weight = generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 1)))
        @test weight ≈ 2*log(1/10)
        
        new_tr, weight, retdiff, discard = update(tr, (), (), choicemap((:kernel => :idx2, 2)))
                
        # we should still have `:world => :sizes => 1`
        @test new_tr[:world => :sizes => 1 => :size] == tr[:world => :sizes => 1 => :size]
        @test new_tr[:kernel => :idx2] == 2
        
        @test weight ≈ 0.
        @test ≈(weight, get_score(new_tr) - get_score(tr) - logpdf(normal, new_tr[:world => :sizes => 2 => :size], 2, 1); atol=1e-10)
        
        expected_discard = choicemap(
            (:kernel => :idx2, 1)
        )

        @test discard == expected_discard
    end
end

@testset "simple updates for dynamic gen fn" begin
    simple_update_tests(get_total_size)
end
@testset "simple updates for static gen fn" begin
    simple_update_tests(static_get_total_size)
end
