module DataAnalysis
using JSON
using Serialization
using Plots
using Dates
include("../types.jl")
include("entropy_metric.jl")
include("../data/import_data.jl")

function get_state(filename)
    f = open(filename, "r")
    st = read_state(f)
    close(f)
    st
end

function calculate_entropies_over_time(ground_truth_filename, datadir)
    filenames = [name for name in readdir(datadir, sort=true)]
    ground_truth = get_state(ground_truth_filename)

    times = []
    entropies_of_groundtruth_given_inferred = []
    entropies_of_inferred_given_groundtruth = []
    for filename in filenames
        state = get_state(joinpath(datadir, filename))


        split_filename = split(filename, "---")
        if length(split_filename) < 2
            println("Found filename $filename not matching the expected pattern; skipping.")
            continue
        end
        datetime = DateTime(split_filename[2], "yyyymmdd-HH_MM_SS-sss")
        push!(times, datetime)

        push!(entropies_of_groundtruth_given_inferred, entropy_of_second_given_first(state, ground_truth))
        push!(entropies_of_inferred_given_groundtruth, entropy_of_second_given_first(ground_truth, state))

    end
    mintime = min(times...)
    runtimes = [Dates.value(time - mintime)/1000 for time in times] # times in seconds since first recorded iteration

    return (runtimes, entropies_of_inferred_given_groundtruth, entropies_of_groundtruth_given_inferred)
end

function plot_inf_given_gt_entropies(groundtruth_filepath, labels_to_paths...)
    plot_entropies(groundtruth_filepath, labels_to_paths...; inf_given_gt=true, title="Entropy of inferred relations given ground truth")
end
function plot_gt_given_inf_entropies(groundtruth_filepath, labels_to_paths...)
    plot_entropies(groundtruth_filepath, labels_to_paths...; inf_given_gt=false, title="Entropy of ground truth relations given inferences")
end

function plot_entropies(groundtruth_filepath, labels_to_paths...; inf_given_gt=true, title="Entropies over time")
    labels_to_paths = collect(labels_to_paths)
    ent_data = [
        calculate_entropies_over_time(groundtruth_filepath, path)
        for (_, path) in labels_to_paths
    ]
    times = [data[1] for data in ent_data]

    i = inf_given_gt ? 2 : 3
    ents = [data[i] for data in ent_data]

    (label1, _) = labels_to_paths[1]
    plot(times[1], ents[1], seriestype = :scatter, label=label1, legend=:bottomleft)
    for i=2:length(labels_to_paths)
        (label, _) = labels_to_paths[i]
        plot!(times[i], ents[i], seriestype = :scatter, label=label)
    end
    plot!(title=title, xlabel="time (s)", ylabel="entropy")
    gui()
end

dirname = joinpath(@__DIR__, "../out/runs/20201010-14_27_00")
gt_dirname = joinpath(dirname, "groundtruth")
labels_to_dirnames = [(l, joinpath(dirname, n)) for (l, n) in (
    "GenWorldModels" => "GenWorldModels", "Vanilla Gen Split/Merge" => "VanillaSplitMerge",
    "Vanilla Gen Generic Infernece" => "VanillaAncestral"
)]
plot_inf_given_gt_entropies(gt_dirname, labels_to_dirnames...)
# path = realpath(joinpath(@__DIR__, "../out/runs/20201010-11_21_22"))
# println("read $path: ", readdir(path))
# plot_entropies(path)
# 20201010-13_53_01
end
