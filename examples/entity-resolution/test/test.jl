module Tests
using Test
using Gen
using GenWorldModels
using FunctionalCollections
import SpecialFunctions: logbeta, loggamma
using Dates
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))
using Distributions: BetaBinomial
using StatsBase: sample

include("../types.jl")
include("../model2/model2.jl")
include("../data/import_data.jl")
include("../inference2/initial_trace.jl")
include("../inference2/inference.jl")

function save_state!(tr, i, dirname)
    f = open(joinpath(dirname, "$i"), "w")
    write_state!(f, get_state(tr))
    close(f)
end

function run_inference_with_saving!(run_inference!, initial_tr, num_iters, examine!, other_args...; save_freq=1, examine_freq=1, kwargs...)
    datetime = Dates.format(now(), "yyyymmdd-HH_MM_SS")
    dirname = joinpath(realpath(joinpath(@__DIR__, "../out/runs")), datetime)
    mkdir(dirname)
    println("Will save to $dirname")

    function saving_examine!(i, tr)
        if i % save_freq === 0
            save_state!(tr, i, dirname)
        end
        examine!(i, tr)
    end

    run_inference!(initial_tr, num_iters, saving_examine!, other_args...; examine_freq=gcd(save_freq, examine_freq), kwargs...)
end

data = read_data()
nents = length(data.entity_strings)
# Idea: calculate the right beta prior!
# avg_num_facts = mean([])
println("num entities: $(length(data.entity_strings))")
params = ModelParams(
    num_entities=length(data.entity_strings),
    num_verbs=length(data.verb_strings),
    num_sentences=length(data.sentences_numeric),
    dirichlet_prior_val=10e-6,
    beta_prior=(0.1, 10.),
    num_relations_mean=length(data.relation_strings),
    num_relations_var=3
)

initial_tr = get_initial_trace(data.sentences_numeric, params)
println("Initial trace generated (score = $(get_score(initial_tr))).\nRunning inference...")

examine!(i, tr) = println("Score at iteration $i : $(get_score(tr)) | num rels = $(tr[:world => :num_relations => ()])")

# inferred = run_tracked_splitmerge_inference!(initial_tr, 1000, examine!, log=true, log_freq=50, examine_freq=100)

run_inference_with_saving!(run_tracked_splitmerge_inference!, initial_tr, 5000, examine!;
    save_freq=50, examine_freq=50, log=true, log_freq=50
)
end