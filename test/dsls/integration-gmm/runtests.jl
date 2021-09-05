using Revise
using Gen
using GenWorldModels
using Test

include("model.jl")
include("inference.jl")

# Params from https://github.com/mugamma/gmm/blob/master/tests/three_component_mixture.jl
# λ, ξ, κ, α_v, β_v = 3, 0.0, 0.01, 2.0, 10.0
# δ = 5.0 (parameter for weight_vec ~ dirichlet(δ))
params = (3, 0.0, 0.01, 2.0, 10.0, 5.0, 1.0)

# Tests:
# include("split_merge_unit_test.jl")
include("three_component_mixture_test.jl")