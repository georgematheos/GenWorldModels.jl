module Types
include("../../types.jl")
end

module GenWorldModelsIE
using Gen
using GenWorldModels
using FunctionalCollections
import SpecialFunctions: logbeta, loggamma
using Dates
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))
using Distributions: BetaBinomial
using StatsBase: sample
using .Types

include("../../model2/model2.jl")
include("../../inference2/initial_trace.jl")
include("../../inference2/inference.jl")



end