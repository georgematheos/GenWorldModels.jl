struct FoldedNormal <: Gen.Distribution{Float64} end
const folded_normal = FoldedNormal()
Gen.is_discrete(::FoldedNormal) = false
Gen.has_output_grad(::FoldedNormal) = false
Gen.has_argument_grads(::FoldedNormal) = (false, false)
function Gen.random(::FoldedNormal, μ, σ)
    abs(random(Gen.normal, μ, σ))
end
function Gen.logpdf(::FoldedNormal, v, μ, σ)
    if v < 0
        -Inf
    else
        Gen.logpdf(Gen.normal, v, μ, σ) + Gen.logpdf(Gen.normal, -v, μ, σ)
    end
end
@dist shifted_folded_normal(minval, μ, σ) = minval + folded_normal(μ, σ)

import Distributions
struct TruncatedNormal <: Gen.Distribution{Float64} end
const truncated_normal = TruncatedNormal()
Gen.is_discrete(::TruncatedNormal) = false
Gen.has_output_grad(::TruncatedNormal) = false
Gen.has_argument_grads(::TruncatedNormal) = (false, false, false, false)
Gen.random(::TruncatedNormal, mu, sigma, l, u) = rand(Distributions.TruncatedNormal(mu, sigma, l, u))
Gen.logpdf(::TruncatedNormal, v, mu, sigma, l, u) = Distributions.logpdf(Distributions.TruncatedNormal(mu, sigma, l, u), v)

struct UnnormalizedCategorical <: Gen.Distribution{Any} end
const unnormalized_categorical = UnnormalizedCategorical()
Gen.is_discrete(::UnnormalizedCategorical) = true
Gen.has_output_grad(::UnnormalizedCategorical) = false
Gen.has_argument_grads(::UnnormalizedCategorical) = (false,)
function Gen.random(::UnnormalizedCategorical, dict)
    if length(dict) == 0
        error("Cannot sample from empty dictionary in unnormalized_categorical!")
    end
    vals_to_scores = collect(dict)
    scores = map(((v, s),) -> s, vals_to_scores)
    vals = map(((v, s),) -> v, vals_to_scores)
    if sum(scores) < 0
        println("sum < 0!")
        if any(x -> x < 0, scores)
            println("  a score value is < 0")
        end
    end
    vals[categorical(scores/sum(scores))]
end
function Gen.logpdf(::UnnormalizedCategorical, val, dict)
    if !haskey(dict, val)
        -Inf
    else
        dict[val] / sum(values(dict))
    end
end