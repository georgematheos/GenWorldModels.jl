module EntityResolution
using Gen
using GenWorldModels

const DIRICHLET_PRIOR_VAL = 0.1
const BETA_PRIOR = (1, 10)
const NUM_REL_PRIOR_MEAN = 2

include("dirichlet.jl")
include("uniform_choice.jl")
include("model.jl")
include("inference.jl")
include("interface.jl")

object_strings = ["cat", "dog"]
verbs = ["chases", "bites", "barks at", "meows at", "runs after"]
sentences = [("dog", "chases", "cat"), ("cat", "bites", "dog")]
sentences_numeric = sentences_string_to_numeric(sentences, object_strings, verbs)

tr = get_initial_trace(sentences_numeric, length(object_strings), length(verbs))
display(get_choices(tr))
display(get_score(tr))

end