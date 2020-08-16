module EntityResolution
using Gen
using GenWorldModels
using FunctionalCollections
using Profile
using ProfileView
using Dates
using Serialization
using SpecialFunctions: loggamma
using JSON
using Distributions: Categorical
using StatsBase: entropy
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))

const DIRICHLET_PRIOR_VAL = 0.2
const BETA_PRIOR = (0.4, 2.)
NUM_REL_PRIOR_MEAN = 3

include("distributions.jl")
include("uniform_choice.jl")
include("emm_counts.jl")
include("model.jl")
include("inference.jl")
include("interface.jl")
include("import_data.jl")
include("evaluate.jl")

function save_num_facts(num_rels, datetime, description, num_true_facts, num_facts_over_time)
    path = joinpath(@__DIR__, "saves", "run_$(datetime)_$(description)_$(num_rels)")
    io = open(path, "w")
    println(io, description)
    println(io, num_rels)
    println(io, num_true_facts)
    println(io, num_facts_over_time)
    close(io)
end

function evaluate_inference()
    datetime = Dates.format(now(), "yyyymmdd-HH_MM_SS")
    num_entities=8
    num_sentences=100
    for num_rels=2:10
        num_verbs=Int(floor(num_rels*1.5))
        global NUM_REL_PRIOR_MEAN = num_rels
        ground_truth_tr = nothing
        while ground_truth_tr === nothing
            try
                ground_truth_tr, _ = generate(generate_sentences, (num_entities, num_verbs, num_sentences), choicemap((:world => :num_relations => () => :num, num_rels)))
            catch e
                println("caught error $e while trying to generate trace for $num_rels; will try again...")
            end
        end
        true_num_facts = length(ground_truth_tr[:kernel => :facts => :facts])

        for splitmergetype in (:sdds, :smart, :dumb)
            println("Running for ", String(splitmergetype), " with $num_rels relations...")
            try
                tr = get_initial_trace(get_retval(ground_truth_tr), num_entities, num_verbs)
                num_facts_over_time = infer_num_facts(tr, splitmergetype, 1000; log_freq=100, examine_freq=10, save_freq=100)
                save_num_facts(num_rels, datetime, String(splitmergetype), true_num_facts, num_facts_over_time)
            catch e
                println("Caught error while running inference for $splitmergetype $num_rels; going to skip this iteration.")
                println("The error was $e")
            end
        end
    end
end
while true
    evaluate_inference()
end

# object_strings = ["cat", "dog"]
# verbs = ["chases", "bites", "barks at", "meows at", "runs after"]
# sentences = [("dog", "chases", "cat"), ("cat", "bites", "dog")]
# sentences_numeric = sentences_string_to_numeric(sentences, object_strings, verbs)

function load_trace_from_save(filename)
    path = joinpath(@__DIR__, "saves", filename)
    constraints = deserialize(path)
    tr,_ = generate(generate_sentences, (length(object_strings), length(verbs), length(sentences_numeric)), constraints)
    return tr
end

# object_strings = ["George", "Alex", "Matin", "MIT", "Berkeley", "CA", "MA"]
# verbs = ["is in", "is within", "studies at", "is a student at", "lives in"]
# sentences = [
#     ("George", "lives in", "MA"),
#     ("George", "is in", "MA"),
#     ("George", "is a student at", "Berkeley"),
#     ("Alex", "lives in", "MA"),
#     ("Alex", "is in", "MA"),
#     ("Alex", "studies at", "MIT"),
#     ("Alex", "is a student at", "MIT"),
#     ("Matin", "lives in", "CA"),
#     ("Matin", "is a student at", "Berkeley"),
#     ("Matin", "studies at", "Berkeley"),
#     ("Berkeley", "is within", "CA"),
#     ("Berkeley", "is in", "CA"),
#     ("MIT", "is in", "MA"),
#     ("George", "lives in", "MA"),
#     ("George", "is in", "MA"),
#     ("George", "is a student at", "Berkeley"),
#     ("Alex", "lives in", "MA"),
#     ("Alex", "is in", "MA"),
#     ("Alex", "studies at", "MIT"),
#     ("Alex", "is a student at", "MIT"),
#     ("Matin", "lives in", "CA"),
#     ("Matin", "is a student at", "Berkeley"),
#     ("Matin", "studies at", "Berkeley"),
#     ("Berkeley", "is within", "CA"),
#     ("Berkeley", "is in", "CA"),
#     ("MIT", "is in", "MA")
# ]

# sentences_numeric = sentences_string_to_numeric(sentences, object_strings, verbs)
# # (sentences, ground_truth_fact_to_indices, idx_to_ground_truth_fact, num_ents, num_verbs) = read_fact_data()
# # evaluate(tr) = evaluate(tr, ground_truth_fact_to_indices, idx_to_ground_truth_fact)

# println("Generating initial trace...")
# tr = get_initial_trace(sentences_numeric, length(object_strings), length(verbs))
# # tr = get_initial_trace(sentences, num_ents, num_verbs)
# println("Initial trace generated; score = ", get_score(tr))
# println()

# println("Beginning inference!")
# mean = infer_mean_num_true_facts(tr, 4000; log_freq=100)
# println("MEAN: $mean")
# # infer_with_benchmarking(tr, 1000; log_freq=10, save_freq=100)

# Profile.clear()
# infer_mean_num_rels(tr, 6; log_freq=1)
# @profile infer_mean_num_rels(tr, 10; log_freq=1)
# ProfileView.view()
# println("MEAN NUM: $mean_num_rels")
end