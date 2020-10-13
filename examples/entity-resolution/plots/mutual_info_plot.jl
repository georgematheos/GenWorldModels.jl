using Serialization
using PyPlot
using Statistics: mean, median

data = deserialize("mutual_informations_full.jld")

# each element of the data vector contains a Dict with:
# "Vanilla Gen generic inference" => (times, mutual_infos)
# "Vanilla Gen split/merge" => (times, mutual_infos)
# "GenWorldModels split/merge" => (times, mutual_infos)

# we want to take the median across runs of the times
# we want to take the mean across runs of the mutual informations

# times = data[i]["Vanilla Gen generic inference"][1]

function median_times(data, alg)
    reps = length(data)
    num_iters = length(data[1][alg][1])
    results = Float64[]
    for iter in 1:num_iters
        push!(results, median([run[alg][1][iter] for run in data]))
    end
    return results
end

function mean_mutual_infos(data, alg)
    reps = length(data)
    num_iters = length(data[1][alg][1])
    means = Float64[]
    for iter in 1:num_iters
        push!(means, mean([run[alg][2][iter] for run in data]))
    end
    return means
end

function plot_series(data, alg, label, color, linestyle)
    times = median_times(data, alg)
    plot(
        sort(times),
        mean_mutual_infos(data, alg)[sortperm(times)],
        label=label, color=color, linestyle=linestyle)
end

figure(figsize=(4,3), dpi=300)
plot_series(data, "Vanilla Gen generic inference", "Vanilla Gen, Generic", "red", "--")
plot_series(data, "Vanilla Gen split/merge", "Vanilla Gen, Split/Merge", "red", "-")
plot_series(data, "GenWorldModels split/merge", "Ours, Split/Merge", "blue", "-")
legend()
gca().set_ylim(0, 4.8)
xlabel("Time (seconds)")
ylabel("Mutual Information (nats)")
tight_layout()
savefig("mutual_info.pdf")
