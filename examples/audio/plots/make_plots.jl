using PyPlot
using Statistics: mean, median, quantile, std

# defines:
# - generic_times
# - bd_times
# - sm_times
# - generic_log_likelihoods
# - bd_log_likelihoods
# - sm_log_likelihoods
include("data.jl")

function medians(data)
    reps = length(data)
    n = length(data[1])
    results = Float64[]
    for i in 1:n
        push!(results, median([data[j][i] for j in 1:length(data)]))
    end
    return results
end

function means(data)
    reps = length(data)
    n = length(data[1])
    means = Float64[]
    for i in 1:n
        push!(means, mean([data[j][i] for j in 1:length(data)]))
    end
    return means
end

function iqr_upper(data)
    reps = length(data)
    n = length(data[1])
    results = Float64[]
    for i in 1:n
        push!(results, quantile([data[j][i] for j in 1:length(data)], 0.75))
    end
    return results
end

function iqr_lower(data)
    reps = length(data)
    n = length(data[1])
    results = Float64[]
    for i in 1:n
        push!(results, quantile([data[j][i] for j in 1:length(data)], 0.25))
    end
    return results
end

function std_errors(data)
    reps = length(data)
    n = length(data[1])
    results = Float64[]
    for i in 1:n
        push!(results, std([data[j][i] for j in 1:length(data)]) / sqrt(reps))
    end
    return results
end

function plot_series(times, log_likelihoods, color, label)
    plot(
        times,
        log_likelihoods,
        label=label, color=color)
end

function error_bars(times, upper, lower, color)
    fill_between(
        times,
        lower, upper, 
        alpha=0.2,
        color=color)
end

close("all")

# mean log likelihoods, with standard errors
figure(figsize=(4,4))
plot_series(medians(generic_times), means(generic_log_likelihoods), "green", "generic kernel")
plot_series(medians(sm_times), means(sm_log_likelihoods), "orange", "split/merge + birth/death kernel")
plot_series(medians(bd_times), means(bd_log_likelihoods), "blue", "birth/death kernel")
error_bars(
    medians(generic_times),
    means(generic_log_likelihoods) .- std_errors(generic_log_likelihoods),
    means(generic_log_likelihoods) .+ std_errors(generic_log_likelihoods), "green")
error_bars(
    medians(sm_times),
    means(sm_log_likelihoods) .- std_errors(sm_log_likelihoods),
    means(sm_log_likelihoods) .+ std_errors(sm_log_likelihoods), "orange")
error_bars(
    medians(bd_times),
    means(bd_log_likelihoods) .- std_errors(bd_log_likelihoods),
    means(bd_log_likelihoods) .+ std_errors(bd_log_likelihoods), "blue")
gca().set_xlim(0, 25)
gca().set_ylim(-300000, 0)
tight_layout()
xlabel("time (s)")
ylabel("Average log likelihood with standard error")
title("Comparing inference accuracy for different algorithms")
legend(loc="lower right")
savefig("means.pdf")

# median log likelihoods, with IQR
figure(figsize=(4,4))
plot_series(medians(generic_times), medians(generic_log_likelihoods), "green", "generic kernel")
plot_series(medians(sm_times), means(sm_log_likelihoods), "orange", "split/merge + birth/death kernel")
plot_series(medians(bd_times), means(bd_log_likelihoods), "blue", "birth/death kernel")
error_bars(medians(generic_times), iqr_lower(generic_log_likelihoods), iqr_upper(generic_log_likelihoods), "green")
error_bars(medians(sm_times), iqr_lower(sm_log_likelihoods), iqr_upper(sm_log_likelihoods), "orange")
error_bars(medians(bd_times), iqr_lower(bd_log_likelihoods), iqr_upper(bd_log_likelihoods), "blue")
gca().set_xlim(0, 25)
gca().set_ylim(-300000, 0)
tight_layout()
xlabel("time (s)")
ylabel("Median log likelihood with IQR")
title("Comparing inference accuracy for different algorithms")
legend(loc="lower right")
savefig("medians.pdf")
