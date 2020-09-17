struct BetaBernoulliSubsetChoiceMap{S, SS} <: Gen.AddressTree{Value}
    set::S
    subset::SS
end
Gen.get_subtree(c::BetaBernoulliSubsetChoiceMap, a) = Value(a in c.subset)
Gen.get_subtrees_shallow(c::BetaBernoulliSubsetChoiceMap) = ((a, Value(a in c.subset)) for a in c.set)

struct BetaBernoulliSubsetTrace <: Gen.Trace
    args::Tuple{AbstractArray, Float64, Float64}
    subset::PersistentSet
    logpdf::Float64
end
Gen.get_retval(tr::BetaBernoulliSubsetTrace) = tr.subset
Gen.get_choices(tr::BetaBernoulliSubsetTrace) = BetaBernoulliSubsetChoiceMap(tr.args[1], tr.subset)
Gen.get_score(tr::BetaBernoulliSubsetTrace) = tr.logpdf
Gen.project(tr::BetaBernoulliSubsetTrace, ::EmptyAddressTree) = 0.
Gen.get_args(tr::BetaBernoulliSubsetTrace) = tr.args
Gen.get_gen_fn(::BetaBernoulliSubsetTrace) = beta_bernoulli_subset

struct BetaBernoulliSubset <: Gen.GenerativeFunction{PersistentSet{Int}, BetaBernoulliSubsetTrace} end

"""
    beta_bernoulli_subset(set, α, β)

Generates a subset of `set` by sampling the inclusion of
each set element from a beta/bernoulli process.

`set` should be an abstract array (not a Julia Set).
"""
beta_bernoulli_subset = BetaBernoulliSubset()

function Gen.generate(::BetaBernoulliSubset, (set, α, β)::Tuple{AbstractArray, Real, Real}, constraints::ChoiceMap)
    certainly_true = Set()
    certainly_false = Set()
    for (element, truthiness) in get_values_shallow(constraints)
        truthiness ? push!(certainly_true, element) : push!(certainly_false, element)
    end

    num_known = length(certainly_false) + length(certainly_true)
    num_additional_true = rand(BetaBinomial(length(set) - num_known, α, β))

    # if the number new samples we need is significantly less than the number of options,
    # do rejection sampling (ie. draw a random sample & reject if we've already drawn it)
    if num_additional_true*5 < length(set)-num_known
        samples = copy(certainly_true)
        for _=1:num_additional_true
            s = sample(set)
            no_go = 
            while s in samples || s in certainly_false
                s = sample(set)
            end
            push!(samples, s)
        end
    else
        # if we have a lot of samples, the rejection rate would be high, so use this
        # slower algorithm from StatsBase.  (statsbase does this dispatch too, but it's slow
        # to construct this list of elements removing the certainly trues and falses)
        spliced = collect(setdiff(Set(set), certainly_true, certainly_false))
        samples = sample(spliced, num_additional_true, replace=false)
    end
    
    trues = PersistentSet(certainly_true)
    for sample in samples
        trues = push(trues, sample)
    end

    num_true = length(trues)
    num_false = length(set) - num_true
    base_lbeta = logbeta(α, β)
    score = logbeta(α + num_true, β + num_false) - base_lbeta
    weight = logbeta(α + length(certainly_true), β + length(certainly_false)) - base_lbeta

    tr = BetaBernoulliSubsetTrace((set, α, β), trues, score)
    (tr, weight)
end

function Gen.update(
    tr::BetaBernoulliSubsetTrace,
    args::Tuple{AbstractArray, Real, Real},
    ::Tuple{NoChange, NoChange, NoChange},
    constraints::ChoiceMap,
    ::Selection
)
    new_subset = tr.subset
    added = Set(); deleted = Set()
    discard = choicemap()
    for (element, put_in_set) in get_values_shallow(constraints)
        if put_in_set && !(element in get_retval(tr))
            new_subset = push(new_subset, element)
            push!(added, element)
            discard[element] = false
        elseif !put_in_set && (element in get_retval(tr))
            new_subset = disj(new_subset, element)
            push!(deleted, element)
            discard[element] = true
        end
    end

    new_num_true = length(new_subset)
    old_num_true = length(tr.subset)
    total = length(tr.args[1])
    (α, β) = args[2:3]
    weight = logbeta(α + new_num_true, β + total - new_num_true) - logbeta(α + old_num_true, β + total - old_num_true)
    new_tr = BetaBernoulliSubsetTrace(args, new_subset, get_score(tr) + weight)

    (new_tr, weight, SetDiff(added, deleted), discard)
end