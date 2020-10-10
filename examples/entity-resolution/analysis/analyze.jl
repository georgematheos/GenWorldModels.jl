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

function calculate_entropies_over_time(datadir)
    filenames = [name for name in readdir(datadir, sort=true)]
    ground_truth = get_state(joinpath(datadir, "groundtruth"))

    times = []
    entropies_of_groundtruth_given_inferred = []
    entropies_of_inferred_given_groundtruth = []
    for filename in filenames
        if filename == "groundtruth"
            continue
        end
        state = get_state(joinpath(datadir, filename))

        push!(entropies_of_groundtruth_given_inferred, entropy_of_second_given_first(state, ground_truth))
        push!(entropies_of_inferred_given_groundtruth, entropy_of_second_given_first(ground_truth, state))

        datetime = DateTime(split(filename, "---")[2], "yyyymmdd-HH_MM_SS-sss")
        push!(times, datetime)
    end
    mintime = min(times...)
    runtimes = [Dates.value(time - mintime)/1000 for time in times] # times in seconds since first recorded iteration

    return (runtimes, entropies_of_inferred_given_groundtruth, entropies_of_groundtruth_given_inferred)
end

function plot_entropies(path)
    (times, entropies_of_inferred_given_groundtruth, entropies_of_groundtruth_given_inferred) = calculate_entropies_over_time(path)
    plot(times, entropies_of_groundtruth_given_inferred, seriestype = :scatter, label="entropy of groundtruth given inferred", legend=:bottomleft)
    plot!(times, entropies_of_inferred_given_groundtruth, seriestype = :scatter, label="entropy of inferred given groundtruth")
    plot!(title="Entropies throughout inference", xlabel="time (s)", ylabel="entropy")
    gui()
end

path = realpath(joinpath(@__DIR__, "../out/runs/20201010-11_21_22"))
println("read $path: ", readdir(path))
plot_entropies(path)
end
