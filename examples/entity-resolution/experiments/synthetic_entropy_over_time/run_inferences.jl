module RunInferences
module Types
include("../../types.jl")
export ModelParams, SentencesNumeric, SentenceNumeric, write_params!, write_state!, FactNumeric, State
end

using Gen
using GenWorldModels
using FunctionalCollections
import SpecialFunctions: logbeta, loggamma
using Dates
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))
using Distributions: BetaBinomial
using StatsBase: sample
using .Types

# include("../../types.jl")
include("../../model2/model2.jl")
include("../../inference2/initial_trace.jl")
include("../../inference2/inference.jl")

function save_state!(state, i, currenttime, dirname; filename=nothing)
    if filename === nothing
        filename = "iter-$i--time---" * Dates.format(currenttime, "yyyymmdd-HH_MM_SS-sss")
    end
    f = open(joinpath(dirname, filename), "w")
    write_state!(f, state)
    close(f)
end
function save_params!(params, dirname; filename="params")
    f = open(joinpath(dirname, filename), "w")
    write_params!(f, params)
    close(f)
end

function run_inference_with_saving!(run_inference!, initial_tr, num_iters, examine!, other_args...; dirname, get_state=get_state, save_freq=1, examine_freq=1, kwargs...)
    function saving_examine!(i, tr)
        if i % save_freq === 0
            save_state!(get_state(tr), i, Dates.now(), dirname)
        end
        examine!(i, tr)
    end

    # save_state!(get_state(initial_tr), 0, Dates.now(), dirname)
    run_inference!(initial_tr, num_iters, saving_examine!, other_args...; examine_freq=gcd(save_freq, examine_freq), kwargs...)
end

function get_groundtruth_tr(params)
    tr, _ = generate(generate_sentences, model_args(params), EmptyAddressTree(); check_proper_usage=false)
    num_true_facts = length(tr[:kernel => :sampled_facts => :all_facts])
    num_sampled_facts = length(unique(tr[:kernel => :sampled_facts => :sampled_facts]))
    println("Num true: ", num_true_facts, "; num sampled: ", num_sampled_facts)
    tr, _ = generate(generate_sentences, model_args(params), EmptyAddressTree(); check_proper_usage=false)
    num_true_facts = length(tr[:kernel => :sampled_facts => :all_facts])
    num_sampled_facts = length(unique(tr[:kernel => :sampled_facts => :sampled_facts]))
    return tr
end

function get_sentences_numeric_from_tr(tr)
    return [
        SentenceNumeric(verb=v, ent1=entpair[1], ent2=entpair[2])
        for (v, entpair) in zip(get_retval(tr)...)
    ]
end

params = ModelParams(
    num_entities=50, #200,
    num_verbs=120,
    num_sentences=1000,
    dirichlet_prior_val=0.01,
    beta_prior=(20, 10000),
    num_relations_mean=70,
    num_relations_var=60
)

groundtruth_tr = get_groundtruth_tr(params)
println("Ground truth trace generated.")
println("\tScore: ", get_score(groundtruth_tr))
println("\tNum true facts: ", length(groundtruth_tr[:kernel => :sampled_facts => :all_facts]))
println("\tNum sampled facts: ", length(unique(groundtruth_tr[:kernel => :sampled_facts => :sampled_facts])))
println("\tNum relations: ", groundtruth_tr[:world => :num_relations => ()])

datetime = Dates.format(now(), "yyyymmdd-HH_MM_SS")
dirname = joinpath(realpath(joinpath(@__DIR__, "../../out/runs")), datetime)
mkdir(dirname)
println("Will save to $dirname")

save_params!(params, dirname)
save_state!(get_state(groundtruth_tr), nothing, nothing, dirname; filename="groundtruth")

##################
# GenWorldModels #
##################
println("Running GenWorldModels split/merge...")
gwm_dirname = joinpath(dirname, "GenWorldModels")
mkdir(gwm_dirname)
examine(i, tr) = println("Score at iteration $i : $(get_score(tr)) | num rels = $(tr[:world => :num_relations => ()])")

initial_tr = get_initial_trace(get_sentences_numeric_from_tr(groundtruth_tr), params)
println("Initial trace generated (score = $(get_score(initial_tr))).\nRunning inference...")

run_inference_with_saving!(run_tracked_splitmerge_inference!, initial_tr, 1200, examine;
dirname=gwm_dirname, save_freq=4, examine_freq=10, log=true, log_freq=10
)

###########################
# Vanilla Gen split/merge #
###########################
println("Running Vanilla Gen split/merge...")
vsm_dirname = joinpath(dirname, "VanillaSplitMerge")
mkdir(vsm_dirname)
examine(i, tr) = println("Score at iteration $i : $(get_score(tr)) | num rels = $(tr[:num_rels])")

include("../../vanilla/main.jl")
initial_tr = VanillaIE.get_trace_for_state(get_state(initial_tr), VanillaIE.get_sentences_numeric_from_tr(groundtruth_tr), params)
println("Initial trace generated (score = $(get_score(initial_tr))).\nRunning inference...")
run_inference_with_saving!(VanillaIE.run_tracked_splitmerge_inference!, initial_tr, 125, examine;
dirname=vsm_dirname, get_state=VanillaIE.get_state, save_freq=1, examine_freq=10, log=true, log_freq=10
)

##################################
# Vanilla Gen Ancestral Sampling #
##################################
println("Running Vanilla Gen ancestral sampling...")
va_dirname = joinpath(dirname, "VanillaAncestral")
mkdir(va_dirname)

run_inference_with_saving!(VanillaIE.run_ancestral_sampling!, initial_tr, 1800, examine;
dirname=va_dirname, get_state=VanillaIE.get_state, save_freq=10, examine_freq=10
)


end