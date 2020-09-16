module DataAnalysis
using JSON
using Serialization
using Plots
include("../types.jl")
include("entropy_metric.jl")
include("../data/import_data.jl")

function get_state(filename)
    f = open(filename, "r")
    st = read_state(f)
    close(f)
    st
end

function calculate_entropies(datadir)
    data = read_data()
    nrels = length(data.relation_strings)
    sentence_rels = map(fact -> fact.rel, data.facts_numeric)
    ground_truth = State(nrels, Set{FactNumeric}(), sentence_rels)    

    filenames = [name for name in readdir(datadir, sort=true) if tryparse(Int, name) !== nothing]
    entropies_of_groundtruth_given_inferred = []
    entropies_of_inferred_given_groundtruth = []
    indices = []
    for filename in filenames
        state = get_state(joinpath(datadir, filename))

        push!(entropies_of_groundtruth_given_inferred, entropy_of_second_given_first(state, ground_truth))
        push!(entropies_of_inferred_given_groundtruth, entropy_of_second_given_first(ground_truth, state))
        push!(indices, parse(Int, filename))
        #gui()
    end

    return (entropies_of_groundtruth_given_inferred, entropies_of_inferred_given_groundtruth, indices)
end

function plot_entropies(path)
    (entropies_of_groundtruth_given_inferred, entropies_of_inferred_given_groundtruth, indices) = calculate_entropies(path)

    plot(indices, entropies_of_groundtruth_given_inferred, seriestype = :scatter, label="entropy of groundtruth given inferred", legend=:bottomleft)
    plot!(indices, entropies_of_inferred_given_groundtruth, seriestype = :scatter, label="entropy of inferred given groundtruth")
    plot!(title="Entropies throughout inference", xlabel="num iterations run", ylabel="entropy")
    gui()
end

path = realpath(joinpath(@__DIR__, "../out/runs/20200916-15_46_09"))
println("read $path: ", readdir(path))
plot_entropies(path)
end
