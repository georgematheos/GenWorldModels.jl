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

function save_state!(tr, i, currenttime, dirname; filename=nothing)
    if filename === nothing
        filename = "iter-$i--time---" * Dates.format(currenttime, "yyyymmdd-HH_MM_SS-sss")
    end
    f = open(joinpath(dirname, filename), "w")
    write_state!(f, get_state(tr))
    close(f)
end

function run_inference_with_saving!(run_inference!, initial_tr, num_iters, examine!, other_args...; groundtruth=nothing, save_freq=1, examine_freq=1, kwargs...)
    datetime = Dates.format(now(), "yyyymmdd-HH_MM_SS")
    dirname = joinpath(realpath(joinpath(@__DIR__, "../out/runs")), datetime)
    mkdir(dirname)
    println("Will save to $dirname")

    if groundtruth !== nothing
        save_state!(groundtruth, -1, Dates.now(), dirname; filename="groundtruth")
    end

    function saving_examine!(i, tr)
        if i % save_freq === 0
            save_state!(tr, i, Dates.now(), dirname)
        end
        examine!(i, tr)
    end

    save_state!(initial_tr, 0, Dates.now(), dirname)
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
    num_sentences=length(data.sentences_numeric)*6,
    dirichlet_prior_val=0.01,
    beta_prior=(20, 10000),
    num_relations_mean=length(data.relation_strings)*3*4,
    num_relations_var=100
)

# initial_tr = get_initial_trace(data.sentences_numeric, params)
function get_groundtruth_tr(params)
    tr, _ = generate(generate_sentences, model_args(params), EmptyAddressTree(); check_proper_usage=false)
    num_true_facts = length(tr[:kernel => :sampled_facts => :all_facts])
    num_sampled_facts = length(unique(tr[:kernel => :sampled_facts => :sampled_facts]))
    # for i=1:100
    #     if num_true_facts - num_sampled_facts <= 4
    #         return tr    
    #     end
        println("Num true: ", num_true_facts, "; num sampled: ", num_sampled_facts)
        tr, _ = generate(generate_sentences, model_args(params), EmptyAddressTree(); check_proper_usage=false)
        num_true_facts = length(tr[:kernel => :sampled_facts => :all_facts])
        num_sampled_facts = length(unique(tr[:kernel => :sampled_facts => :sampled_facts]))
        return tr
    # end
    # error()
end

function get_sentences_numeric_from_tr(tr)
    return [
        SentenceNumeric(verb=v, ent1=entpair[1], ent2=entpair[2])
        for (v, entpair) in zip(get_retval(tr)...)
    ]
end

groundtruth_tr = get_groundtruth_tr(params)
println("Ground truth trace generated.")
println("\tScore: ", get_score(groundtruth_tr))
println("\tNum true facts: ", length(groundtruth_tr[:kernel => :sampled_facts => :all_facts]))
println("\tNum sampled facts: ", length(unique(groundtruth_tr[:kernel => :sampled_facts => :sampled_facts])))
println("\tNum relations: ", groundtruth_tr[:world => :num_relations => ()])

initial_tr = get_initial_trace(get_sentences_numeric_from_tr(groundtruth_tr), params)
println("Initial trace generated (score = $(get_score(initial_tr))).\nRunning inference...")

function get_examine()
    num_ones = 0
    function examine!(i, tr)
        # if i > 150
            println("Score at iteration $i : $(get_score(tr)) | num rels = $(tr[:world => :num_relations => ()])")
        # end
        # if tr[:world => :num_relations => ()] == 1
        #     num_ones += 1
        #     if num_ones >= 3
        #         error("too many visits to a state with only one relation!")
        #     end
        # end
    end
    return examine!
end

# inferred = run_tracked_splitmerge_inference!(initial_tr, 1000, examine!, log=true, log_freq=50, examine_freq=100)
success = false
# for _=1:10
#     # success && break
    # try
            # run_inference_with_saving!(run_tracked_splitmerge_inference!, initial_tr, 10000, get_examine();
            #     save_freq=50, examine_freq=100, log=true, log_freq=100, groundtruth=groundtruth_tr
            # )
            run_tracked_splitmerge_inference!(initial_tr, 2000, get_examine(), log=true, log_freq=100, examine_freq=100)
        println("Just finished running...")
        success = true
#     catch e
#     #     if e isa ErrorException && occursin("key not found", e.msg)
#     #         for (exc, bt) in Base.catch_stack()
#     #             showerror(stdout, exc, bt)
#     #             println()
#     #         end
#     #     end
#         println("Breaking due to error $e")
#     end
# end
end