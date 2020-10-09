struct BetaBernoulliSubsetChoiceMap{S, SS} <: Gen.AddressTree{Value}
    set::S
    subset::SS
end
Gen.get_subtree(c::BetaBernoulliSubsetChoiceMap, a) = Value(a in c.subset)
Gen.get_subtree(::BetaBernoulliSubsetChoiceMap, ::Pair) = EmptyAddressTree()
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
function Gen.project(tr::BetaBernoulliSubsetTrace, sel::Selection)
    num_selected_true = 0
    num_selected_false = 0
    for (addr, subsel) in get_subtrees_shallow(sel)
        subsel isa EmptyAddressTree && continue;
        @assert subsel isa AllSelection
        if addr in tr.subset
            num_selected_true += 1
        else
            num_selected_false += 1
        end
    end

    ar, α, β = tr.args
    # num_true = length(tr.subset)
    # num_false = length(ar) - num_true
    # pt = logbeta(α + num_true, β + num_false) - logbeta(α, β)
    # prob_gend = logbeta(α + num_true, β + num_false) - logbeta(α + num_selected_true, β + num_selected_false)
    # println("PROJECT returning Β(α + |S|, β)/Β(α, β)")
    return logbeta(α + num_selected_true, β + num_selected_false) - logbeta(α, β) # = pt - prob_gend
end
Gen.get_args(tr::BetaBernoulliSubsetTrace) = tr.args
Gen.get_gen_fn(::BetaBernoulliSubsetTrace) = beta_bernoulli_subset

struct BetaBernoulliSubset <: Gen.GenerativeFunction{PersistentSet, BetaBernoulliSubsetTrace} end

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
    # println("Given $(length(certainly_true)) true, $(length(certainly_false)) false; sampled $num_additional_true more true.")

    # if the number new samples we need is significantly less than the number of options,
    # do rejection sampling (ie. draw a random sample & reject if we've already drawn it)
    if num_additional_true*5 < length(set)-num_known
        samples = copy(certainly_true)
        for _=1:num_additional_true
            s = sample(set)
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
    # println("GENERATE returning Β(α + |S|, β)/Β(α, β)")

    tr = BetaBernoulliSubsetTrace((set, α, β), trues, score)
    (tr, weight)
end

# TODO: DRY
function Gen.generate(::BetaBernoulliSubset, (set, α, β)::Tuple{AbstractArray, Real, Real}, constraints::Value)
    trues = get_value(constraints)
    num_true = length(trues)
    num_false = length(set) - num_true
    base_lbeta = logbeta(α, β)
    score = logbeta(α + num_true, β + num_false) - base_lbeta
    # println("GENERATE returning Β(α + |T|, β + E^2 - |T|)/Β(α, β)")

    tr = BetaBernoulliSubsetTrace((set, α, β), trues, score)
    (tr, score)
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

    # println("UPDATE!!")
    (new_tr, weight, SetDiff(added, deleted), discard)
end

function Gen.update(
    tr::BetaBernoulliSubsetTrace,
    args::Tuple{AbstractArray, Real, Real},
    ::Tuple{Gen.NoChange,Gen.NoChange,Gen.NoChange},
    constraints::Value,
    ::AllSelection
)
    new_tr, weight = generate(get_gen_fn(tr), args, constraints)
    @assert isapprox(weight, get_score(new_tr))
    return (new_tr, get_score(new_tr) - get_score(tr), UnknownChange(), get_choices(tr))
end