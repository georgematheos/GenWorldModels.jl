module LargeRealData
using Gen
using GenWorldModels
using FunctionalCollections
import SpecialFunctions: logbeta, loggamma
using Dates
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))
using Distributions: BetaBinomial
using StatsBase: sample

include("../../types.jl")
include("../../model2/model2.jl")
include("../../data/import_unlabelled_data.jl")
include("../../inference2/initial_trace.jl")
include("../../inference2/inference.jl")

function get_params(filename, num; mean_n_rels=100)
    data = read_data(filename, num=num)
    nents = length(data.entity_strings)
    println("Num entities: $nents")
    println("Num sentences: $(length(data.sentences_numeric))")
    
    println("Setting mean num relations to $mean_n_rels")
    
    params = ModelParams(
        num_entities=length(data.entity_strings),
        num_verbs=length(data.verb_strings),
        num_sentences=length(data.sentences_numeric),
        dirichlet_prior_val=10e-6,
        beta_prior=(0.02, .18),
        num_relations_mean=mean_n_rels,
        num_relations_var=50
    )

    return (params, data.sentences_numeric)
end



# println("Going to generate initial trace...")
# @time initial_tr = get_initial_trace(data.sentences_numeric, params)
# println("Initial trace generated (score = $(get_score(initial_tr))).\nRunning inference...")


end