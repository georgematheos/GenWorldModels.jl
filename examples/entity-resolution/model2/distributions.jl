import Distributions
struct Dirichlet <: Distribution{Vector{<:Real}} end
const dirichlet = Dirichlet()
Gen.random(::Dirichlet, α) = rand(Distributions.Dirichlet(α))
Gen.logpdf(::Dirichlet, v, α) = Distributions.logpdf(Distributions.Dirichlet(α), v)
dirichlet(α) = random(dirichlet, α)

struct CategoricalFromList <: Gen.Distribution{Any} end
const categorical_from_list = CategoricalFromList()
function Gen.random(::CategoricalFromList, list, probs)
    idx = categorical(probs)
    list[idx]
end
function Gen.logpdf(::CategoricalFromList, v, list, probs)
    idxs = findall(x -> x == v, list)
    @assert length(idxs) == 1
    idx = idxs[1]
    log(probs[idx])
end

struct UniformFromList <: Gen.Distribution{Any} end
const uniform_from_list = UniformFromList()
function Gen.random(::UniformFromList, list)
    idx = uniform_discrete(1, length(list))
    return list[idx]
end
function Gen.logpdf(::UniformFromList, obj, list)
    if obj in list
        -log(length(list))
    else
        -Inf
    end
end

struct DiscreteLogNormal <: Gen.Distribution{Int} end
const discrete_log_normal = DiscreteLogNormal()
Gen.random(::DiscreteLogNormal, μ, σ) = floor(rand(Distributions.LogNormal(μ, σ)))
function Gen.logpdf(::DiscreteLogNormal, x, μ, σ)
    dist = Distributions.LogNormal(μ, σ)
    log(Distributions.cdf(dist, x+1) - Distributions.cdf(dist, x))
end

struct UniformChoice <: Gen.Distribution{Any} end
const uniform_choice = UniformChoice()
function Gen.random(::UniformChoice, set)
    lst = collect(set)
    lst[uniform_discrete(1, length(set))]
end
Gen.logpdf(::UniformChoice, val, set) = val in set ? -log(length(set)) : -Inf