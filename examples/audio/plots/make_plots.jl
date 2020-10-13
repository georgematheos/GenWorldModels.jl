using PyPlot
using Statistics: mean, median, quantile, std

# defines:
# - generic_times
# - bd_times
# - sm_times
# - generic_likelihoods
# - bd_likelihoods
# - sm_likelihoods
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

function plot_series(times, log_likelihoods, color, label, linestyle="-")
    plot(
        times,
        log_likelihoods,
        label=label, color=color, linestyle=linestyle)
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
figure(figsize=(5,3), dpi=300)
plot_series(medians(generic_times), means(generic_likelihoods), "black", "Generic kernel", "--")
plot_series(medians(sm_times), means(sm_likelihoods), "green", "Split/Merge + Birth/Death Kernel", "-")
plot_series(medians(bd_times), means(bd_likelihoods), "magenta", "Birth/Death Kernel", "-")
#error_bars(
    #medians(generic_times),
    #means(generic_likelihoods) .- std_errors(generic_likelihoods),
    #means(generic_likelihoods) .+ std_errors(generic_likelihoods), "black")
#error_bars(
    #medians(sm_times),
    #means(sm_likelihoods) .- std_errors(sm_likelihoods),
    #means(sm_likelihoods) .+ std_errors(sm_likelihoods), "green")
#error_bars(
    #medians(bd_times),
    #means(bd_likelihoods) .- std_errors(bd_likelihoods),
    #means(bd_likelihoods) .+ std_errors(bd_likelihoods), "blue")
gca().set_xlim(0, 25)
gca().set_ylim(-200000, 0)
tight_layout()
xlabel("Time (seconds)")
ylabel("Average log likelihood of data")
legend(loc="lower right")
tight_layout()
savefig("means.png")

# median log likelihoods, with IQR
figure(figsize=(5,3))
plot_series(medians(generic_times), medians(generic_likelihoods), "black", "Generic Kernel")
plot_series(medians(sm_times), means(sm_likelihoods), "green", "Split/Merge + Birth/Death Kernel")
plot_series(medians(bd_times), means(bd_likelihoods), "magenta", "Birth/Death Kernel")
error_bars(medians(generic_times), iqr_lower(generic_likelihoods), iqr_upper(generic_likelihoods), "black")
error_bars(medians(sm_times), iqr_lower(sm_likelihoods), iqr_upper(sm_likelihoods), "green")
error_bars(medians(bd_times), iqr_lower(bd_likelihoods), iqr_upper(bd_likelihoods), "blue")
gca().set_xlim(0, 25)
gca().set_ylim(-300000, 0)
tight_layout()
xlabel("time (s)")
ylabel("Median log likelihood with IQR")
title("MCMC convergence")
legend(loc="lower right")
savefig("medians.pdf")
