module Memoization
using Gen
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
    @testset "basic generate" begin
        diff_idx_tr, _ = generate(get_total_size, (), choicemap((:idx1, 1), (:idx2, 2), (:sizes => 1 => :size, 1.05)))
        same_idx_tr, _ = generate(get_total_size, (), choicemap((:idx1, 1), (:idx2, 1), (:sizes => 1 => :size, 1.05)))

        @test length(get_submaps_shallow(get_submap(get_choices(diff_idx_tr), :sizes))) == 2
        @test length(get_submaps_shallow(get_submap(get_choices(same_idx_tr), :sizes))) == 1
        @test diff_idx_tr[:sizes => 1 => :size] == 1.05
        @test diff_idx_tr[:sizes => 2 => :size] != 1.05 # this should be a 0 probability event
        @test same_idx_tr[:sizes => 1 => :size] == 1.05

        @test get_score(same_idx_tr) ≈ log(1/10) + log(1/10) + logpdf(normal, 1.05, 1, 1)
        @test get_score(diff_idx_tr) ≈ get_score(same_idx_tr) + logpdf(normal, diff_idx_tr[:sizes => 2 => :size], 2, 1)
    end
    @testset "erroring generate" begin
        @test_throws Exception generate(get_total_size, (), choicemap((:idx1, 1), (:idx2, 2), (:sizes => 3 => :size, 3.05)))
    end
end

end # module