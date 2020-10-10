module RealIE
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

function get_params(;
    filename="nyt8500.json",
    num=nothing,
    mean_n_rels=100,
    num_relations_var=70,
    beta_prior=nothing,
    dirichlet_prior_val=0.01
)
    data = read_data(filename, num=num)
    nents = length(data.entity_strings)
    println("Num entities: $nents")
    println("Num sentences: $(length(data.sentences_numeric))")
    
    println("Setting mean num relations to $mean_n_rels")

    if beta_prior === nothing
        entpairs = ((sent.ent1, sent.ent2) for sent in data.sentences_numeric)
        num_entpairs = length(unique(entpairs))
        println("num entpairs: ", num_entpairs)
        avg_num_true = num_entpairs/mean_n_rels
        beta_prior = (avg_num_true, nents^2 - avg_num_true)
    end
    println("beta prior: $beta_prior")
    
    params = ModelParams(
        num_entities=length(data.entity_strings),
        num_verbs=length(data.verb_strings),
        num_sentences=length(data.sentences_numeric),
        dirichlet_prior_val=dirichlet_prior_val,
        beta_prior=beta_prior,
        num_relations_mean=mean_n_rels,
        num_relations_var=num_relations_var
    )

    return (params, data.sentences_numeric)
end



# println("Going to generate initial trace...")
# @time initial_tr = get_initial_trace(data.sentences_numeric, params)
# println("Initial trace generated (score = $(get_score(initial_tr))).\nRunning inference...")


end