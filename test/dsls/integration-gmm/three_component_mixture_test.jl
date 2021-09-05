mean(vector) = sum(vector) / length(vector)
value_constraints(vals) = choicemap(((@obsmodel() => :vals => i, val) for (i, val) in enumerate(vals))...)

# Matin's smart initial trace generation function:
function initial_trace_smart(ys, k, iters=3)
    # K-means for a small number of iterations
    n = length(ys)
    μs = ys[[uniform_discrete(1, n) for _=1:k]]
    zs = [uniform_discrete(1, k) for _=1:n]
    recenter(j) = let y_js = ys[zs .== j];
                      isempty(y_js) ? ys[uniform_discrete(1, n)] : mean(y_js) end
    for _=1:iters
        μs = map(recenter, 1:k)
        zs = map(y->argmin((y .- μs).^2), ys)
    end
    σ²s = map(j->mean((ys[zs .== j] .- μs[j]).^2), 1:k)
    σ²s[σ²s .== 0.0] .= 1.0
    tr, = generate(
        gaussian_mixture_model,
        (params..., length(ys)),
        merge(value_constraints(ys), choicemap(
            @set_number(Cluster(), 3),
            (@set(mean[Cluster(i)] => :mean, val) for (i, val) in enumerate(μs))...,
            (@set(var[Cluster(i)] => :var, val) for (i, val) in enumerate(σ²s))...,
            ((@obsmodel() => :cluster_samples => i, Cluster(val)) for (i, val) in enumerate(zs))...,
        ))
    );
    return tr
end

@testset "correctly infers three component mixture" begin
    ys = [11.26, 28.93, 30.52, 30.09, 29.46, 10.03, 11.24, 11.55, 30.4, -18.44,
          10.91, 11.89, -20.64, 30.59, 14.84, 13.54, 7.25, 12.83, 11.86, 29.95,
          29.47, -18.16, -19.28, -18.87, 9.95, 28.24, 9.43, 7.38, 29.46, 30.73,
          7.75, 28.29, -21.99, -20.0, -20.86, 15.5, -18.62, 13.11, 28.66,
          28.18, -18.78, -20.48, 9.18, -20.12, 10.2, 30.26, -14.94, 5.45, 31.1,
          30.01, 10.52, 30.48, -20.37, -19.3, -21.92, -18.31, -18.9, -20.03,
          29.32, -17.53, 10.61, 6.38, -20.72, 10.29, 11.21, -18.98, 8.57,
          10.47, -22.4, 6.58, 29.8, -17.43, 7.8, 9.72, -21.53, 11.76, 29.72,
          29.31, 6.82, 15.51, 10.69, 29.56, 8.84, 30.93, 28.75, 10.72, 9.21,
          8.57, 11.92, -23.96, -19.78, -17.2, 11.79, 29.95, 7.29, 6.57, -17.99,
          13.29, -22.53, -20.0]
    zs = [2, 3, 3, 3, 3, 2, 2, 2, 3, 1, 2, 2, 1, 3, 2, 2, 2, 2, 2, 3, 3, 1, 1,
          1, 2, 3, 2, 2, 3, 3, 2, 3, 1, 1, 1, 2, 1, 2, 3, 3, 1, 1, 2, 1, 2, 3,
          1, 2, 3, 3, 2, 3, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1,
          2, 3, 1, 2, 2, 1, 2, 3, 3, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 2, 1, 1, 1,
          2, 3, 2, 2, 1, 2, 1, 1]
    @assert length(ys) == length(zs)

    initial_tr = initial_trace_smart(ys, 2, 3);
    @test get_score(initial_tr) > -Inf
    println("Initial trace generation successful.")

    inferred_tr = do_inference(initial_tr, 300; get_map = true);

    true_means = [-20., 10., 30.]
    true_vars = [3.0, 5.0, 1.0]

    mean_error(i, mean) = abs(@get(inferred_tr, mean[Cluster(i)]) - mean)
    closest_idx_to_mean(mean) = argmin([mean_error(i, mean) for i=1:3])
    idxs = [closest_idx_to_mean(mean) for mean in true_means]

    for (idx, mean, var) in zip(idxs, true_means, true_vars)
        @test mean_error(idx, mean) < 1.0
        @test abs(@get(inferred_tr, var[Cluster(idx)]) - var) < 2.0
    end

    inferred_idx(inferred_tr, test_to_world_idx, datapoint_idx) = findfirst([Cluster(i) == inferred_tr[@obsmodel() => :cluster_samples => datapoint_idx] for i in test_to_world_idx])
    n_misallocated = sum([zs[datapoint_idx] != inferred_idx(inferred_tr, idxs, datapoint_idx) for datapoint_idx=1:length(ys)])
    println("$n_misallocated / $(length(zs)) samples misallocated")
    @test n_misallocated < 0.05 * length(ys)
end
