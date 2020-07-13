using GenWorldModels: @w, @w_args, @UsingWorld
# TODO: test @WorldMap

@gen function s1(world, sample_idx)
    @w_args mean1, mean2
    which ~ bernoulli(0.5)
    mean = which ? mean1 : mean2
    return {:val} ~ normal(mean, 1)
end

@gen (static) function s2(world, sample_idx)
    @w_args mean1
    val ~ normal(mean1, 1)
    return val
end

@gen (static) function _draw_sample_sum(world, total_num_samples)
    i ~ uniform_discrete(1, total_num_samples)
    sample1 ~ @w s1[i]
    sample2 ~ @w s2[i]
    return sample1 + sample2
end

draw_sample_sum = @UsingWorld(_draw_sample_sum, s1, s2; world_args=(:mean1, :mean2))

@load_generated_functions()

@testset "macros" begin
    tr = simulate(draw_sample_sum, (1., -1., 4))
    tr, weight = generate(draw_sample_sum, (0.5, 2.54, 2), choicemap((:kernel => :i, 1), (:world => :s2 => 1 => :val, 1.)))
    @test tr[:kernel => :i] == 2
    @test tr[:world => :s2 => 1] == 1.
    @test isapprox(weight, log(1/2) + logpdf(normal, 1., 2.54, 1))
end