module Tests

using Test
using Gen
using FunctionalCollections
import SpecialFunctions: logbeta, loggamma
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))
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

include("../model2/dirichlet_process_entity_mention.jl")
@testset "dirichlet_process_entity_mention" begin
    tr, wt = generate(dirichlet_process_entity_mention, (["a", "a", "b"], [0.2, 0.2]))
    @test wt == 0.
    @test get_retval(tr) isa AbstractVector
    @test length(get_retval(tr)) == 3

    tr, wt = generate(dirichlet_process_entity_mention, (["a", "a", "b"], [0.2, 0.2]), choicemap((1, 1), (2, 1), (3, 2)))
    exp_score = logbeta([2.2,.2]) - logbeta([.2,.2]) + logbeta([.2, 1.2]) - logbeta([.2,.2])
    @test get_retval(tr) == [1, 1, 2]
    @test isapprox(get_score(tr), exp_score)
    @test isapprox(wt, exp_score)

    new_tr, weight, retdiff, discard = update(tr, (["a", "b", "b"], [0.2, 0.2]), (
            VectorDiff(3, 3, Dict(2 => UnknownChange())), NoChange()
        ),
        EmptyChoiceMap()
    )
    @test get_retval(new_tr) == get_retval(tr)
    @test retdiff === NoChange()
    @test isempty(discard)

    exp_score = logbeta([1.2,1.2]) - logbeta([.2,.2]) + logbeta([1.2, .2]) - logbeta([.2,.2])
    @test isapprox(get_score(new_tr), exp_score)
    @test isapprox(weight, get_score(new_tr) - get_score(tr))

    ents = PersistentVector([rand() < .6 ? rand() < .5 ? "a" : "b" : "c" for _=1:1000])
    α = [.5, .5, .5, .5]
    tr, wt = generate(
        dirichlet_process_entity_mention,
        (ents, α)
    )
    ments = get_retval(tr)
    cnts = [[length([i for i=1:1000 if ents[i] == ent && ments[i] == j]) for j=1:4] for ent in ["a", "b", "c", "d"]]
    exp_score = sum(logbeta(cnt + α) for cnt in cnts) - 4*logbeta(α)
    @test isapprox(get_score(tr), exp_score)
    @test wt == 0.

    changed_ents = collect(2:2:102)
    for idx in changed_ents
        ents = assoc(ents, idx, ents[idx] == "a" ? "b" : "a")
    end
    new_tr, weight, retdiff, discard = update(tr, (ents, α), (
        VectorDiff(1000, 1000, Dict(idx => UnknownChange() for idx in changed_ents)), NoChange()), EmptyAddressTree()
    )
    @test get_retval(new_tr) == get_retval(tr)
    ments = get_retval(new_tr)
    cnts = [[length([i for i=1:1000 if ents[i] == ent && ments[i] == j]) for j=1:4] for ent in ["a", "b", "c", "d"]]
    println("expected counts: $cnts")
    exp_score = sum(logbeta(cnt + α) for cnt in cnts) - 4*logbeta(α)
    @test isapprox(get_score(new_tr), exp_score)
    @test isapprox(weight, get_score(new_tr) - get_score(tr))
    @test retdiff === NoChange()
    @test isempty(discard)
end

end