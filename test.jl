module Memoization
using Gen
using FunctionalCollections
using DataStructures
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

@testset "simple UsingMemoized function" begin
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

@testset "scene decoration UsingMemoized" begin
    struct Object
        size::Float64
        color::Int
    end
    
    @gen (static, diffs) function generate_object(idx)
        size ~ normal(idx, 1)
        color ~ uniform_discrete(1, 6)
        obj = Object(size, color)
        return obj
    end
    
    @gen (static, diffs) function sample_object(num_objects, objects)
        idx ~ uniform_discrete(1, num_objects)
        obj ~ lookup(objects, idx)
        return obj
    end
    sample_objects = Map(sample_object)
    
    @gen (static, diffs) function generate_scene_kernel(objects::LookupTable)
        num_objects ~ poisson(30)
        num_in_scene ~ poisson(3)
        
        scene ~ sample_objects(fill(num_objects, num_in_scene), fill(objects, num_in_scene))
        return scene
    end
    Gen.@load_generated_functions()
    
    generate_scene = UsingMemoized(generate_scene_kernel, :objects => generate_object)
    
    tr, weight = generate(generate_scene, (), choicemap(
        (:kernel => :num_in_scene, 3),
        (:kernel => :num_objects, 30),
        (:kernel => :scene => 1 => :idx, 1),
        (:kernel => :scene => 2 => :idx, 2),
        (:kernel => :scene => 3 => :idx, 3),
    ))
    @test weight ≈ logpdf(poisson, 3, 3) + logpdf(poisson, 30, 30) + 3*log(1/30)
    
    obj_gen_scores = [log(1/6) + logpdf(normal, tr[:objects => i => :size], i, 1) for i=1:3]
    @test get_score(tr) ≈ weight + sum(obj_gen_scores)
    
    new_tr, weight, retdiff, discard = update(tr, (), (), choicemap((:kernel => :num_in_scene, 2)))
    @test get_submap(get_choices(new_tr), :objects => 3) == EmptyChoiceMap()
    @test weight ≈ logpdf(poisson, 2, 3) - logpdf(poisson, 3, 3) - log(1/30) - (log(1/6) + logpdf(normal, tr[:objects => 3 => :size], 3, 1))
    @test get_score(new_tr) - get_score(tr) ≈ weight
end

create_tuple(a, b) = (a, b)
create_tuple(a::Diffed, b::Diffed) = Diffed((strip_diff(a), strip_diff(b)), UnknownChange())

@gen function approx_factorial(tup)
    n, fact_lookup = tup
    if n == 0
        return {:val} ~ normal(1, 0.05)
    else
        f_nminus1 ~ lookup(fact_lookup, create_tuple(n-1, fact_lookup))
        return {:val} ~ normal(n * f_nminus1, 0.05)
    end
end

@gen function sample_approx_factorial_kernel(fact_lookup, n)
    val ~ lookup(fact_lookup, create_tuple(n, fact_lookup))
    return val
end

const sample_approx_factorial = UsingMemoized(sample_approx_factorial_kernel, :factorial => approx_factorial)
Gen.@load_generated_functions()

@testset "passing in lookuptable in idx" begin
    tr, weight = generate(sample_approx_factorial, (4,))
    
    # gotta get the lookup table object to check stuff:
    choices = get_choices(tr)
    
    lookup_idx_addr, (n, lt) = collect(filter(((key, val),) -> key != :val, get_values_shallow(get_submap(choices, :kernel => :val))))[1]

    @test n == 4
    fact_vals = [tr[:factorial => (i, lt) => :val] for i=0:4]
    
    fact_val_scores = [logpdf(normal, fact_vals[1], 1, 0.05)]
    for i=1:4
        push!(fact_val_scores, logpdf(normal, fact_vals[i + 1], i * fact_vals[i], 0.05))
    end
    
    @test get_score(tr) ≈ sum(fact_val_scores)

    # now we're gonna update the trace. this should result in entirely resampling
    # the factorial values, since the `fact_lookup` object passed in is a new object,
    # so every `(n, fact_lookup)` index is a new index
    # another way to look at it is that the `q` for `UsingMemoized` always proposes that
    # the `LookupTable` objects are changed, and thus these indices change
    # (the contract with the kernel that the lookup table behave like a concrete lookup table
    # is never broken, since for all the kernel knows, a new lookup table with new
    # unused values in it is passed in)
    new_tr, weight, retdiff, discard = update(tr, (4,), (NoChange(),), EmptyChoiceMap())
    
    lookup_idx_addr, (n, lt_new) = collect(filter(((key, val),) -> key != :val, get_values_shallow(get_submap(get_choices(new_tr), :kernel => :val))))[1]
    @test n == 4
    
    # test the lookup tables are different
    @test lt_new != lt
    
    # test the approx factorial values are different now
    for i=0:4
        @test lookup(lt_new, (i, lt_new)) != lookup(lt, (i, lt))
        @test !__has_value__(lt_new, (i, lt))
        @test get_submap(discard, :factorial => (i, lt)) != EmptyChoiceMap()
    end
    @test get_retval(new_tr) != get_retval(tr)
    
    fact_vals = [new_tr[:factorial => (i, lt_new) => :val] for i=0:4]
    
    fact_val_scores = [logpdf(normal, fact_vals[1], 1, 0.05)]
    for i=1:4
        push!(fact_val_scores, logpdf(normal, fact_vals[i + 1], i * fact_vals[i], 0.05))
    end
    
    @test get_score(new_tr) ≈ sum(fact_val_scores)
    @test weight ≈ -get_score(tr)
end

@gen function approx_factorial2(n, approx_fact_lookup)
    if n == 0
        return {:val} ~ normal(1, 0.05)
    else
        f_nminus1 ~ lookup(approx_fact_lookup, n - 1)
        return {:val} ~ normal(n * f_nminus1, 0.05)
    end
end

@gen function approx_fact2_kern(approx_fact_lookup, n)
    val ~ lookup(approx_fact_lookup, n)
    return val
end

approx_fact_2 = UsingMemoized(approx_fact2_kern, :factorial => (approx_factorial2, :factorial))
Gen.@load_generated_functions()

@testset "simple recursion on dynamic gen fn" begin
    tr, weight = generate(approx_fact_2, (4,))

    fact_vals = [tr[:factorial => i => :val] for i=0:4]
    fact_val_scores = [logpdf(normal, fact_vals[1], 1, 0.05)]
    for i=1:4
        push!(fact_val_scores, logpdf(normal, fact_vals[i + 1], i * fact_vals[i], 0.05))
    end
    @test get_score(tr) ≈ sum(fact_val_scores)
    @test weight == 0.
    
    new_tr, weight, retdiff, discard = update(tr, (4,), (NoChange(),), choicemap((:factorial => 0 => :val, 1.00)))
    
    @test get_submap(discard, :factorial => 0) != EmptyChoiceMap()
    
    # since the dynamic memoized gen fn can't track retdiffs, it will end up discarding
    # all 5 submaps of :factorial.  If it could track retdiffs, it would only discard 2:
    # for index 0, whose value changes, and for index 1, which has in it's choicemap a lookup
    # to index 0, whose value changed.  However, indices 2,3,4 will get a NoChange
    # input retdiff and won't discard anything
    @test length(get_submaps_shallow(get_submap(discard, :factorial))) == 5
    
    fact_vals = [new_tr[:factorial => i => :val] for i=0:4]
    fact_val_scores = [logpdf(normal, fact_vals[1], 1, 0.05)]
    for i=1:4
        push!(fact_val_scores, logpdf(normal, fact_vals[i + 1], i * fact_vals[i], 0.05))
    end
    @test get_score(new_tr) ≈ sum(fact_val_scores)
    @test weight ≈ get_score(new_tr) - get_score(tr)
end

@gen function approx_fib_recursive(n, fib_lookup)
    if n == 0 || n == 1
        val ~ normal(1, 0.05)
    else
        prev1 ~ lookup(fib_lookup, n - 1)
        prev2 ~ lookup(fib_lookup, n - 2)
        sum = prev1 + prev2
        val ~ normal(sum, 0.05)
    end
    return val
end

@gen (static, diffs) function calc_approx_fib_kernel(fib_lookup, n)
    val ~ lookup(fib_lookup, n)
    return val
end
Gen.@load_generated_functions()
approx_fib = UsingMemoized(calc_approx_fib_kernel, :fib => (approx_fib_recursive, :fib))

@testset "approximate fib recursively" begin
    tr, weight = generate(approx_fib, (6,), choicemap((:fib => 2 => :val, 2.000)))
    @test tr[:fib => 2 => :val] == 2.000
    @test weight ≈ logpdf(normal, 2.000, tr[:fib => 0] + tr[:fib => 1], 0.05)
    
    # I'm putting these in a constraints weird order (idk if it effects what order `get_submaps_shallow` returns things in, though)
    # but in any case update(::UsingMemoized) should topologically order the constraints so everything ends up being fine
    new_tr, weight, _, discard = update(tr, (6,), (NoChange(),), choicemap(
        (:fib => 0 => :val, 1.05), (:fib => 2 => :val, 2.01), (:fib => 1 => :val, 0.95)
    ))
    @test new_tr[:fib => 2] == 2.01
    @test new_tr[:fib => 0] == 1.05
    @test new_tr[:fib => 1] == 0.95
    @test new_tr[:fib => 3] == tr[:fib => 3]
    @test new_tr[:fib => 4] == tr[:fib => 4]
    
    # scores for indices 0 through 4 are effected by this update
    expected_weight = (
          logpdf(normal, new_tr[:fib => 0], 1, 0.05) - logpdf(normal, tr[:fib => 0], 1, 0.05)
        + logpdf(normal, new_tr[:fib => 1], 1, 0.05) - logpdf(normal, tr[:fib => 1], 1, 0.05)
        + logpdf(normal, new_tr[:fib => 2], new_tr[:fib => 0] + new_tr[:fib => 1], 0.05) - logpdf(normal, tr[:fib => 2], tr[:fib => 0] + tr[:fib => 1], 0.05)
        + logpdf(normal, new_tr[:fib => 3], new_tr[:fib => 1] + new_tr[:fib => 2], 0.05) - logpdf(normal, tr[:fib => 3], tr[:fib => 1] + tr[:fib => 2], 0.05)
        + logpdf(normal, new_tr[:fib => 4], new_tr[:fib => 2] + new_tr[:fib => 3], 0.05) - logpdf(normal, tr[:fib => 4], tr[:fib => 2] + tr[:fib => 3], 0.05)
    )
    @test weight ≈ expected_weight
    @test weight ≈ get_score(new_tr) - get_score(tr)
end

@gen (static, diffs) function a(idx, b_lookup)
    b_val ~ lookup(b_lookup, idx)
    val ~ normal(b_val, 0.05)
    return val
end

@gen function b(idx, a_lookup)
    if idx == 1
        val ~ normal(0, 1)
    else
        a_val ~ lookup(a_lookup, idx - 1)
        val ~ normal(a_val, 1)
    end
    return val
end

@gen (static, diffs) function ab_kernel(a_lookup, b_lookup)
    idx ~ uniform_discrete(1, 4)
    a_val ~ lookup(a_lookup, idx)
    b_val ~ lookup(b_lookup, idx)
    sum = a_val + b_val
    return sum
end

Gen.@load_generated_functions()

ab = UsingMemoized(ab_kernel, :a => (a, :b), :b => (b, :a))

@testset "interdependent memoized gen functions" begin
    tr, weight = generate(ab, (), choicemap((:kernel => :idx, 4)))
    @test weight ≈ log(1/4)
    
    new_tr, weight, retdiff, discard = update(tr, (), (), choicemap(
        (:kernel => :idx, 3),
        (:a => 1 => :val, 0.0),
        (:b => 2 => :val, 1.0)
    ))
    
    @test new_tr[:a => 1] == 0.0
    @test new_tr[:b => 2] == 1.0
    @test new_tr[:b => 1] == tr[:b => 1]
    @test new_tr[:a => 3] == tr[:a => 3]

    new_choices = get_choices(new_tr)
    @test get_submap(new_choices, :a => 4) == EmptyChoiceMap()
    @test get_submap(new_choices, :b => 4) == EmptyChoiceMap()
    
    # a1, b2 changed. a2 depends on b2 so it's weight must change. a4, b4 deleted. other lookups shouldn't change
    expected_weight = (
        logpdf(normal, new_tr[:a => 1], new_tr[:b => 1], 0.05) - logpdf(normal, tr[:a => 1], tr[:b => 1], 0.05)
      + logpdf(normal, new_tr[:b => 2], new_tr[:a => 1], 1) - logpdf(normal, tr[:b => 2], tr[:a => 1], 1)
      + logpdf(normal, new_tr[:a => 2], new_tr[:b => 2], 0.05) - logpdf(normal, tr[:a => 2], tr[:b => 2], 0.05)
      - logpdf(normal, tr[:b => 4], tr[:a => 3], 1) - logpdf(normal, tr[:a => 4], tr[:b => 4], 0.05)
    )
    @test weight ≈ expected_weight
    @test weight ≈ get_score(new_tr) - get_score(tr)
end

end # module

