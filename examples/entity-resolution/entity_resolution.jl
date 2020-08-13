module EntityResolution
using Gen
using GenWorldModels
using FunctionalCollections
using Profile
using ProfileView
using Dates
using Serialization

const DIRICHLET_PRIOR_VAL = 0.5
const BETA_PRIOR = (1, 10)
const NUM_REL_PRIOR_MEAN = 2

include("dirichlet.jl")
include("uniform_choice.jl")
include("emm_counts.jl")
include("model.jl")
include("inference.jl")
include("interface.jl")

object_strings = ["cat", "dog"]
verbs = ["chases", "bites", "barks at", "meows at", "runs after"]
sentences = [("dog", "chases", "cat"), ("cat", "bites", "dog")]
sentences_numeric = sentences_string_to_numeric(sentences, object_strings, verbs)

function load_trace_from_save(filename)
    path = joinpath(@__DIR__, "saves", filename)
    constraints = deserialize(path)
    tr,_ = generate(generate_sentences, (length(object_strings), length(verbs), length(sentences_numeric)), constraints)
    return tr
end

# tr = get_initial_trace(sentences_numeric, length(object_strings), length(verbs))
# display(get_choices(tr))
# display(get_score(tr))

# println("Beginning inference!")
# mean = infer_mean_num_rels(tr, 1; log_freq=1, save_freq=1)
# println("MEAN: $mean")

# Profile.clear()
# infer_mean_num_rels(tr, 6; log_freq=1)
# @profile infer_mean_num_rels(tr, 10; log_freq=1)
# ProfileView.view()
# println("MEAN NUM: $mean_num_rels")
end