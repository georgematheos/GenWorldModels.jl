module VanillaIE
using Gen
using GenWorldModels
using FunctionalCollections
import SpecialFunctions: logbeta, loggamma
using Dates
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))
using Distributions: BetaBinomial
using StatsBase: sample
using ..Types

# include("../types.jl")
include("model.jl")
include("initial_trace.jl")
include("inference.jl")
include("../data/import_unlabelled_data.jl")

function get_groundtruth_tr(params)
    tr, _ = generate(generate_sentences, model_args(params), EmptyAddressTree())
    num_true_facts = length(tr[:sampled_facts => :all_facts])
    num_sampled_facts = length(unique(tr[:sampled_facts => :sampled_facts]))
    println("Num true: ", num_true_facts, "; num sampled: ", num_sampled_facts)
    return tr
end
function get_sentences_numeric_from_tr(tr)
    return [
        SentenceNumeric(verb=v, ent1=entpair[1], ent2=entpair[2])
        for (v, entpair) in zip(get_retval(tr)...)
    ]
end

function get_params(;
    filename="nyt8500.json",
    num=nothing,
    mean_n_rels=100,
    num_relations_var=70,
    beta_prior=nothing,
    dirichlet_prior_val=0.001
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

# params = ModelParams(
#     num_entities=2,
#     num_verbs=4,
#     num_sentences=30,
#     dirichlet_prior_val=0.001,
#     beta_prior=(1, 1),
#     num_relations_mean=3,
#     num_relations_var=1
# )

# groundtruth_tr = get_groundtruth_tr(params)
# println("Ground truth trace generated.")
# println("\tScore: ", get_score(groundtruth_tr))

# initial_tr = get_initial_trace(get_sentences_numeric_from_tr(groundtruth_tr), params)
# println("Initial trace generated (score = $(get_score(initial_tr))).\nRunning inference...")

# examine(i, tr) = println("Score at iteration $i : $(get_score(tr)) | num rels = $(tr[:num_rels])")

# run_tracked_splitmerge_inference!(initial_tr, 3000, examine, log=true, log_freq=30, examine_freq=30)

# println("Inference completed.")
end