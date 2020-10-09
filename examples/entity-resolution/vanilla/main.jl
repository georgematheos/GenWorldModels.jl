module VanillaIE
using Gen
using GenWorldModels
using FunctionalCollections
import SpecialFunctions: logbeta, loggamma
using Dates
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))
using Distributions: BetaBinomial
using StatsBase: sample

include("../types.jl")
include("model.jl")
include("initial_trace.jl")
include("inference.jl")

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

params = ModelParams(
    num_entities=2,
    num_verbs=4,
    num_sentences=30,
    dirichlet_prior_val=0.001,
    beta_prior=(1, 1),
    num_relations_mean=3,
    num_relations_var=1
)

groundtruth_tr = get_groundtruth_tr(params)
println("Ground truth trace generated.")
println("\tScore: ", get_score(groundtruth_tr))

initial_tr = get_initial_trace(get_sentences_numeric_from_tr(groundtruth_tr), params)
println("Initial trace generated (score = $(get_score(initial_tr))).\nRunning inference...")

examine(i, tr) = println("Score at iteration $i : $(get_score(tr)) | num rels = $(tr[:num_rels])")

run_tracked_splitmerge_inference!(initial_tr, 100, examine, log=true, log_freq=1, examine_freq=1)

println("Inference completed.")
end