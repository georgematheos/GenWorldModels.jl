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
        
        # we should throw an error if we don't use all the constraints we provide to generate
        # TODO: mabe we should change this behavior
        @test_throws Exception generate(get_total_size, (), choicemap((:kernel => :idx1, 1), (:kernel => :idx2, 2), (:world => :sizes => 3 => :size, 3.05)))
       
        # test error throwing behavior when we don't trace world accesses properly
        @test_throws Exception generate(broken_get_total_size1, ())
        @test_throws Exception generate(broken_get_total_size2, ())
        @test_throws Exception generate(broken_get_total_size3, ())
    end
end

end