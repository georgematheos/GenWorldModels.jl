module Memoization
using Gen
using FunctionalCollections
using Test

include("lookup_table.jl")
include("lookup.jl")
include("using_memoized.jl")

@gen (static, diffs) function generate_size(idx)
    size ~ normal(idx, 1)
    return size
end

@gen (static, diffs) function kernel(sizes::LookupTable)    
    idx1 ~ uniform_discrete(1, 10)
    idx2 ~ uniform_discrete(1, 10)
    
    size1 ~ lookup(sizes, idx1)
    size2 ~ lookup(sizes, idx2)
    
    total = size1 + size2
    return total
end

const get_total_size = UsingMemoized(kernel, :sizes => generate_size)

Gen.load_generated_functions()

@testset "kernel on ConcreteLookupTable" begin
    sizes = Dict(i => normal(i, 1) for i=1:10)
    sizes = ConcreteLookupTable(sizes)
    
    tr, weight = generate(kernel, (sizes,))

    @test weight == 0.
    @test get_score(tr) ≈ log(1/10) + log(1/10)
    @test tr[:size1 => :val] == lookup(sizes, tr[:idx1])
    @test tr[:size2 => :val] == lookup(sizes, tr[:idx2])
end

@testset "UsingMemoized function" begin
    # should throw error if we try to use address :kernel
    @test_throws Exception UsingMemoized(kernel, :kernel => generate_size)
    
    @testset "simple generate" begin
        diff_idx_tr, diff_weight = generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 2), (:sizes => 1 => :size, 1.05)))
        same_idx_tr, same_weight = generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 1), (:sizes => 1 => :size, 1.05)))

        @test length(get_submaps_shallow(get_submap(get_choices(diff_idx_tr), :sizes))) == 2
        @test length(get_submaps_shallow(get_submap(get_choices(same_idx_tr), :sizes))) == 1
        @test diff_idx_tr[:sizes => 1 => :size] == 1.05
        @test diff_idx_tr[:sizes => 2 => :size] != 1.05 # this should be a 0 probability event
        @test same_idx_tr[:sizes => 1 => :size] == 1.05

        @test get_score(same_idx_tr) ≈ log(1/10) + log(1/10) + logpdf(normal, 1.05, 1, 1)
        @test get_score(diff_idx_tr) ≈ get_score(same_idx_tr) + logpdf(normal, diff_idx_tr[:sizes => 2 => :size], 2, 1)
        @test same_weight ≈ get_score(same_idx_tr) # this should have been fully constrained
        @test diff_weight ≈ same_weight
        
        @test_throws Exception generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 2), (:sizes => 3 => :size, 3.05)))
    end
    
    @testset "simple updates" begin
        tr, weight = generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 2)))
        @test weight ≈ 2*log(1/10)
        
        @testset "update memoized values without changing lookups" begin
            new_tr, weight, retdiff, discard = update(tr, (), (), choicemap((:sizes => 1 => :size, 1.000)))
            @test new_tr[:sizes => 1 => :size] == 1.000
            @test new_tr[:kernel => :size1 => :val] == 1.000
            @test weight ≈ logpdf(normal, 1.000, 1, 1) - logpdf(normal, tr[:sizes => 1 => :size], 1, 1)
            @test get_score(new_tr) - get_score(tr) ≈ weight
            
            expected_discard = choicemap(
                (:sizes => 1 => :size, tr[:sizes => 1 => :size]),
                (:kernel => :size1 => :val, tr[:sizes => 1 => :size])
            )
            
            @test discard == expected_discard
        end
        
        @testset "remove lookup" begin
            new_tr, weight, retdiff, discard = update(tr, (), (), choicemap((:kernel => :idx2, 1)))
            
            # there should no longer be a choicemap for :sizes => 2
            @test get_submap(get_choices(new_tr), :sizes => 2) == EmptyChoiceMap()
            
            @test new_tr[:kernel => :size2 => :val] == tr[:kernel => :size1 => :val]
            @test new_tr[:kernel => :size1 => :val] == tr[:kernel => :size1 => :val]
            
            # `size2 ~ lookup` should expose `:val` and another address in the choicemap.
            # this address should give the index which was looked up to get `:val`.
            # check that this has updated to be `1`
            looked_up_idx_addr, looked_up_idx_2 = collect(filter(
               ((key, val),) -> key != :val,
               get_values_shallow(get_submap(get_choices(new_tr), :kernel => :size2))
            ))[1]
            @test looked_up_idx_2 == 1
            @test new_tr[:kernel => :idx2] == 1
            
            @test weight ≈ -logpdf(normal, tr[:sizes => 2 => :size], 2, 1)
            @test get_score(new_tr) - get_score(tr) ≈ weight
            
            expected_discard = choicemap(
                (:kernel => :idx2, 2),
                (:kernel => :size2 => :val, tr[:sizes => 2 => :size]),
                (:kernel => :size2 => looked_up_idx_addr, 2),
                (:sizes => 2 => :size, tr[:sizes => 2 => :size])
            )
            @test discard == expected_discard
        end
        @testset "change lookup" begin
            new_tr, weight, retdiff, discard = update(tr, (), (), choicemap((:kernel => :idx2, 3)))
            
            # there should no longer be a choicemap for :sizes => 2
            @test get_submap(get_choices(new_tr), :sizes => 2) == EmptyChoiceMap()
            
            # we should now have a `:sizes => 3` submap
            @test new_tr[:sizes => 3 => :size] == new_tr[:kernel => :size2 => :val]
            
            looked_up_idx_addr, looked_up_idx_2 = collect(filter(((key, val),) -> key != :val, get_values_shallow(get_submap(get_choices(new_tr), :kernel => :size2))))[1]
            @test looked_up_idx_2 == 3
            @test new_tr[:kernel => :idx2] == 3

            # the weight should simply be the negative logpdf of the dropped value
            # (we don't include the new value in the weight since the choices were totally unconstrained)
            @test weight ≈ -logpdf(normal, tr[:sizes => 2 => :size], 2, 1)
            
            # another way to look at it is that since there is no untraced randomness,
            # exp(weight) = P(new_trace) / (P(old_trace) * Q(generated values | constraints))
            # here, q(generated vals | constraints) = q(:sizes => 3 is given the value it was given)
            @test weight ≈ get_score(new_tr) - get_score(tr) - logpdf(normal, new_tr[:sizes => 3 => :size], 3, 1)
                
            expected_discard = choicemap(
                (:kernel => :idx2, 2),
                (:kernel => :size2 => :val, tr[:sizes => 2 => :size]),
                (:kernel => :size2 => looked_up_idx_addr, 2),
                (:sizes => 2 => :size, tr[:sizes => 2 => :size])
            )
            @test discard == expected_discard
        end
        @testset "add lookup" begin 
            tr, weight = generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 1)))
            @test weight ≈ 2*log(1/10)
            
            new_tr, weight, retdiff, discard = update(tr, (), (), choicemap((:kernel => :idx2, 2)))
            
            # we should now have a `:sizes => 2` submap
            @test new_tr[:sizes => 2 => :size] == new_tr[:kernel => :size2 => :val]
            
            # we should still have `:sizes => 1`
            @test new_tr[:sizes => 1 => :size] == tr[:sizes => 1 => :size]
            @test new_tr[:kernel => :size1 => :val] == new_tr[:sizes => 1 => :size]
            
            looked_up_idx_addr, looked_up_idx_2 = collect(filter(((key, val),) -> key != :val, get_values_shallow(get_submap(get_choices(new_tr), :kernel => :size2))))[1]
            @test looked_up_idx_2 == 2
            @test new_tr[:kernel => :idx2] == 2
            
            @test weight ≈ 0.
            @test ≈(weight, get_score(new_tr) - get_score(tr) - logpdf(normal, new_tr[:sizes => 2 => :size], 2, 1); atol=1e-10)
            
            expected_discard = choicemap(
                (:kernel => :idx2, 1),
                (:kernel => :size2 => :val, tr[:sizes => 1 => :size]),
                (:kernel => :size2 => looked_up_idx_addr, 1)
            )
            @test discard == expected_discard
        end
    end
end

end # module

