module EntityResolution
using Gen
using GenWorldModels
using FunctionalCollections
using Profile
using ProfileView
using Dates
using Serialization
using SpecialFunctions: loggamma
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))

const DIRICHLET_PRIOR_VAL = 0.2
const BETA_PRIOR = (2., 5.)
const NUM_REL_PRIOR_MEAN = 3

include("dirichlet.jl")
include("uniform_choice.jl")
include("emm_counts.jl")
include("model.jl")
include("inference.jl")
include("interface.jl")

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

object_strings = ["George", "Alex", "Matin", "MIT", "Berkeley", "CA", "MA"]
verbs = ["is in", "is within", "studies at", "is a student at", "lives in"]
sentences = [
    ("George", "lives in", "MA"),
    ("George", "is in", "MA"),
    ("George", "is a student at", "Berkeley"),
    ("Alex", "lives in", "MA"),
    ("Alex", "is in", "MA"),
    ("Alex", "studies at", "MIT"),
    ("Alex", "is a student at", "MIT"),
    ("Matin", "lives in", "CA"),
    ("Matin", "is a student at", "Berkeley"),
    ("Matin", "studies at", "Berkeley"),
    ("Berkeley", "is within", "CA"),
    ("Berkeley", "is in", "CA"),
    ("MIT", "is in", "MA")
]

sentences_numeric = sentences_string_to_numeric(sentences, object_strings, verbs)

println("Generating initial trace...")
tr = get_initial_trace(sentences_numeric, length(object_strings), length(verbs))
println("Initial trace generated; score = ", get_score(tr))
println()

println("Beginning inference!")
mean = infer_mean_num_rels(tr, 1000; log_freq=20, save_freq=100)
println("MEAN: $mean")

# Profile.clear()
# infer_mean_num_rels(tr, 6; log_freq=1)
# @profile infer_mean_num_rels(tr, 10; log_freq=1)
# ProfileView.view()
# println("MEAN NUM: $mean_num_rels")
end