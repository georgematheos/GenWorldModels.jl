module Tests
using Test
using Gen
using GenWorldModels
using FunctionalCollections
import SpecialFunctions: logbeta, loggamma
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))
using Distributions: BetaBinomial
using StatsBase: sample

include("../types.jl")
include("../model2/model2.jl")
include("../data/import_data.jl")
include("../inference2/initial_trace.jl")

data = read_data()

params = ModelParams(
    num_entities=length(data.entity_strings),
    num_verbs=length(data.verb_strings),
    num_sentences=length(data.sentences_numeric),
    dirichlet_prior_val=0.9,
    beta_prior=(2, 18),
    num_relations_mean=length(data.relation_strings),
    num_relations_var=3
)

tr = get_initial_trace(data.sentences_numeric, params)

end