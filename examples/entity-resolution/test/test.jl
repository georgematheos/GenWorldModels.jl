module Tests
using Test
using Gen
using GenWorldModels
using FunctionalCollections
import SpecialFunctions: logbeta, loggamma
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))
using Distributions: BetaBinomial
using StatsBase: sample

include("../model2/model2.jl")
tr, _ = generate(generate_sentences, (6, (1, 1), (4, 32), 10, 0.2, 5), EmptyAddressTree(); check_proper_usage=false)

end