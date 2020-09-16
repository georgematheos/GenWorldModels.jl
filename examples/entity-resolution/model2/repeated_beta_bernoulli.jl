function sample_from_beta_binomial(num_samples, α, β)

end

struct RepeatedBetaBernoulliTrace <: Gen.Trace

end

struct RepeatedBetaBernoulli <: Gen.GenerativeFunction{PersistentSet{Int}, RepeatedBetaBernoulliTrace} end
# TODO: I could probably handle this more efficiently either in the case we have a constraint for everything or for nothing
function Gen.generate(::RepeatedBetaBernoulli, (num_samples, α, β)::Tuple{Int, Real, Real}, constraints::ChoiceMap)
    prob = beta(α, β)
    true_vals = Set{Int}()
    for i=1:num_samples
        bernoulli(prob) && push!(true_vals, i)
    end
    unconstrained_trues = copy(true_vals)
    num_constraints = 0
    for (i, valmap) in get_submaps_shallow(constraints)
        isempty(valmap) && continue;
        @assert valmap isa Value
        num_constraints += 1
        delete!(unconstrained_trues, i)
        if get_value(valmap)
            push!(true_vals, i)
        else
            delete!(true_vals, i)
        end
    end

    real_prob = logbeta(α + length(true_vals), β + num_samples)
    unconstrained_prob = logbeta(α + length(unconstrained_trues), β + num_samples - num_constraints)
    tr = RepeatedBetaBernoulliTrace(num_samples, α, β, PersistentSet{Int}(true_vals), real_prob)
    (tr, real_prob - unconstrained_prob)
end

function Gen.update(
    tr::RepeatedBetaBernoulliTrace,
    (num_samples, α, β)::Tuple{Int, Real, Real},
    ::Tuple{NoChange, NoChange, NoChange},
    constraints::ChoiceMap,
    ::Selection
)

end