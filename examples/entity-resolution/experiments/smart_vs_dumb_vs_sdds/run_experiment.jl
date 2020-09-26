### WARNING: I don't think this code is currently working; you probably need
# to see the Git commit when I was running this to get it to work, since I think
# I may have deleted some code it needs.

module SmartVsDumbVsSDDSExperiment
using Gen
using GenWorldModels
using FunctionalCollections
using Dates
using Serialization
using SpecialFunctions: loggamma
using JSON
using Distributions: Categorical
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))

const DIRICHLET_PRIOR_VAL = 0.2
const BETA_PRIOR = (0.4, 2.)
NUM_REL_PRIOR_MEAN = 5

include("../../model/model.jl")
include("../../inference/inference.jl")
include("evaluate_inference_on_synthetic_data.jl")

while true
    evaluate_inference()
end

end