module Tests

using Test
using Gen
using FunctionalCollections
using SpecialFunctions: logbeta
using Distributions: BetaBinomial
using StatsBase: sample
include("../model2/beta_bernoulli_subset.jl")

@testset begin "beta bernoulli subset"
    tr, weight = generate(beta_bernoulli_subset, ([(i, j) for i=1:10, j=1:10], 2, 18))
    @test weight == 0.
    @test get_retval(tr) isa PersistentSet

    set = [(i, j) for i=1:10, j=1:10]
    tr, weight = generate(beta_bernoulli_subset, (set, 20, 10), choicemap(
        ((1, 1), true),
        ((1, 2), true),
        ((1, 3), false)
    ))
    @test (1, 1) in get_retval(tr)
    @test (1, 2) in get_retval(tr)
    @test !((1, 3) in get_retval(tr))

    num_true = length(get_retval(tr))
    num_false = 100 - num_true
    expected_score = logbeta(20 + num_true, 10 + num_false) - logbeta(20, 10)
    @test isapprox(get_score(tr), expected_score)

    expected_weight = get_score(tr) - (logbeta(20 + num_true, 10 + num_false) - logbeta(20 + 2, 10 + 1))
    @test isapprox(weight, expected_weight)

    args = (set, 2, 18)
    tr, weight = generate(beta_bernoulli_subset, args, choicemap(
        ((1, 1), true),
        ((1, 2), true),
        ((1, 3), false)
    ))
    @test (1, 1) in get_retval(tr)
    @test (1, 2) in get_retval(tr)
    @test !((1, 3) in get_retval(tr))

    num_true = length(get_retval(tr))
    num_false = 100 - num_true
    expected_score = logbeta(2 + num_true, 18 + num_false) - logbeta(2, 18)
    @test isapprox(get_score(tr), expected_score)

    expected_weight = get_score(tr) - (logbeta(2 + num_true, 18 + num_false) - logbeta(2 + 2, 18 + 1))
    @test isapprox(weight, expected_weight)

    false_els = [el for el in set if !(el in get_retval(tr))]
    if length(false_els) > 0 && length(get_retval(tr)) > 0
        currently_true = first(get_retval(tr))
        false_el = false_els[1]
        new_tr, weight, retdiff, discard = update(tr, args, (NoChange(), NoChange(), NoChange()), choicemap(
            (false_el, true),
            (currently_true, false)
        ))

        @test false_el in get_retval(new_tr)
        @test !(currently_true in get_retval(new_tr))
        @test isapprox(get_score(new_tr), get_score(tr))
        @test weight == 0.
        @test retdiff == SetDiff(Set([false_el]), Set([currently_true]))
        @test discard == choicemap(
            (false_el, false), (currently_true, true)
        )
    end

    el = sample(set)
    newval = !(el in get_retval(tr))
    new_tr, weight, retdiff, discard = update(tr, args, (NoChange(), NoChange(), NoChange()), choicemap(
            (el, newval)
        ))
    
    @test (el in get_retval(new_tr)) == newval
    num_true = length(get_retval(new_tr))
    num_false = 100 - num_true
    expected_score = logbeta(2 + num_true, 18 + num_false) - logbeta(2, 18)
    @test isapprox(get_score(new_tr), expected_score)

    @test isapprox(weight, get_score(new_tr) - get_score(tr))
    
    @test retdiff == SetDiff(Set([el]), Set())
    @test discard == choicemap(
        (el, !newval)
    )
end

end