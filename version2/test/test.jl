module WorldModelsTests
using Gen
using Test
include("../src/WorldModels.jl")
using .WorldModels

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

@load_generated_functions()

@testset "simple UsingWorld function" begin
    # should throw error if we try to use address :kernel
    @test_throws Exception UsingWorld(kernel, :kernel => generate_size)
    
    @testset "simple generate" begin
        diff_idx_tr, diff_weight = generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 2), (:world => :sizes => 1 => :size, 1.05)))
        same_idx_tr, same_weight = generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 1), (:world => :sizes => 1 => :size, 1.05)))

        @test length(get_submaps_shallow(get_submap(get_choices(diff_idx_tr), :world => :sizes))) == 2
        @test length(get_submaps_shallow(get_submap(get_choices(same_idx_tr), :world => :sizes))) == 1
        @test diff_idx_tr[:world => :sizes => 1 => :size] == 1.05
        @test diff_idx_tr[:world => :sizes => 2 => :size] != 1.05 # this should be a 0 probability event
        @test same_idx_tr[:world => :sizes => 1 => :size] == 1.05

        @test get_score(same_idx_tr) ≈ log(1/10) + log(1/10) + logpdf(normal, 1.05, 1, 1)
        @test get_score(diff_idx_tr) ≈ get_score(same_idx_tr) + logpdf(normal, diff_idx_tr[:world => :sizes => 2 => :size], 2, 1)
        @test same_weight ≈ get_score(same_idx_tr) # this should have been fully constrained
        @test diff_weight ≈ same_weight
        
        @test_throws Exception generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 2), (:world => :sizes => 3 => :size, 3.05)))
    end
end

end