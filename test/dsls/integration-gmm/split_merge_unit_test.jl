function trace_for_which_we_expect_a_split()
    cluster2 = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3]
    cluster1 = [5.0, 5.1, 5.2, 5.05]
    cluster3 = [-3.0, -2.5, -2.8, -2.9, -2.68]
    vals = [cluster1..., cluster2..., cluster3...]
    tr, _ = generate(
        gaussian_mixture_model,
        (params..., length(vals)),
        choicemap(
            @set_number(Cluster(), 2),
            @set(mean[Cluster(1)] => :mean, 0.0), @set(mean[Cluster(2)] => :mean, 5.0),
            @set(var[Cluster(1)] => :var, 1.0), @set(var[Cluster(2)] => :var, 1.0),
            @set(unnormalized_weight[Cluster(1)] => :wt, 1.0), @set(unnormalized_weight[Cluster(2)] => :wt, 1.0),
            (
                (
                    @obsmodel() => :cluster_samples => i,
                    i > length(cluster1) ? Cluster(2) : Cluster(1)
                )
                for i=1:length(vals)
            )...,
            ((@obsmodel() => :vals => i, val) for (i, val) in enumerate(vals))...
        )
    )
    return tr
end


# @testset "gets_split" begin
    tr = trace_for_which_we_expect_a_split()
    any_accepted = false
    for i=1:10
        global tr, acc = mh(tr, MHProposal(split_merge_kernel))
        any_accepted = any_accepted || acc
    end
    println()
    @test any_accepted
    @test @get_number(tr, Cluster()) == 3
# end