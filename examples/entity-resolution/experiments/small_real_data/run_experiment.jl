module SmallRealDataRun
using Gen
using GenWorldModels
using FunctionalCollections
using Dates
using SpecialFunctions: loggamma
using JSON
using Serialization
using Distributions: Categorical
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))

const DIRICHLET_PRIOR_VAL = 0.2
const BETA_PRIOR = (0.2, 1.)
NUM_REL_PRIOR_MEAN = 5

include("../../types.jl")
include("../../model/model.jl")
include("../../inference/inference.jl")
include("../../inference/initial_trace.jl")
include("../../data/import_data.jl")

function save_state!(tr, i, dirname)
    f = open(joinpath(dirname, "$i"), "w")
    write_state!(f, get_state(tr))
    close(f)
end

Tuple(s::SentenceNumeric) = (s.ent1, s.verb, s.ent2)
function run_inference_with_saving!(run_inference!, initial_tr, num_iters, examine!, other_args...; save_freq=1, examine_freq=1, kwargs...)
    datetime = Dates.format(now(), "yyyymmdd-HH_MM_SS")
    dirname = joinpath(realpath(joinpath(@__DIR__, "../../out/runs")), datetime)
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

function run_experiment(data)
    num_rels = length(data.relation_strings)
    sentences = map(Tuple, data.sentences_numeric)
    println("Generating initial trace...")
    initial_tr = get_initial_trace(sentences, length(data.entity_strings), length(data.verb_strings))
    println("Initial trace generated.")

    run_inference_with_saving!(run_tracked_splitmerge_inference!, initial_tr, 1000, (i, tr) -> nothing;
        save_freq=5, examine_freq=5,
        log=true,
        log_freq=10
    )
end

run_experiment(read_data())

end