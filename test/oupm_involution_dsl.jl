@type Sample
@gen (static) function get_val(world::World, sample::Sample)
    val ~ normal(1, 0.5)
    return val
end
@gen (static) function observe_samples_sum_kernel(world::World)
    num_samples ~ poisson(5)
    samples ~ Map(lookup_or_generate)([world[:val][Sample(i)] for i=1:num_samples])
    total = sum(samples)
    observation ~ normal(total, 1)
    return observation
end
@load_generated_functions()
observe_sample_sum = UsingWorld(observe_samples_sum_kernel, :val => get_val; oupm_types=(Sample,))

@testset "OUPM move involution DSL" begin
    OBS = 3.
    tr, _  = generate(observe_sample_sum, (), choicemap(
        (:kernel => :num_samples, 4), (:kernel => :observation, OBS),
    ))

    function run_mh_100(tr, kern, obs)
        new_tr = tr
        for i=1:100
            new_tr, acc = mh(new_tr, kern; check=true, observations=obs)
            println("Iter $i; acc=$acc")
        end
        new_tr
    end
    obs = choicemap((:kernel => :observation, OBS))

    @gen function birth_death_proposal(model_tr)
        do_birth ~ bernoulli(0.5)
        current_num_samples = model_tr[:kernel => :num_samples]
        if do_birth
            idx ~ uniform_discrete(1, current_num_samples + 1)

            current_total = sum(model_tr[:world => :val => Sample(i)] for i=1:current_num_samples)
            expected_val = model_tr[:kernel => :observation] - current_total
            new_val ~ normal(expected_val, 2.)
        else
            idx ~ uniform_discrete(1, current_num_samples)
        end
    end

    # invalid since it doesn't constrain the reverse of a death move!
    # @oupm_involution invalid_bd_involution (old_tr, fwd_prop_tr) to (new_tr, bwd_prop_tr) begin
    #     idx = @read(fwd_prop_tr[:idx], :disc)
    #     current_num_samples = @read(old_tr[:kernel => :num_samples], :disc)
    #     if @read(fwd_prop_tr[:do_birth], :disc)
    #         @birth(Sample, idx)
    #         @write(new_tr[:kernel => :num_samples], current_num_samples + 1, :disc)
    #         new_val = @read(fwd_prop_tr[:new_val], :cont)
    #         @write(new_tr[:world => :val => Sample(idx) => :val], new_val, :cont)
    #         @write(bwd_prop_tr[:do_birth], false, :disc)
    #     else
    #         @death(Sample, idx)
    #         @write(new_tr[:kernel => :num_samples], current_num_samples - 1, :disc)
    #         @write(bwd_prop_tr[:do_birth], true, :disc)
    #     end
    #     @write(bwd_prop_tr[:idx], idx, :disc)
    # end

    # invalid_birth_death_mh_kern = OUPMMHKernel(birth_death_proposal, (), invalid_bd_involution)
    # @test_logs (:error, ) match_mode=:any (@test_throws Exception run_mh_100(tr, invalid_birth_death_mh_kern, obs))

    # now check a valid one!
    println("Real run:")
    @oupm_involution bd_involution (old_tr, fwd_prop_tr) to (new_tr, bwd_prop_tr) begin
        idx = @read(fwd_prop_tr[:idx], :disc)
        do_birth = @read(fwd_prop_tr[:do_birth], :disc)
        current_num_samples = @read(old_tr[:kernel => :num_samples], :disc)
        if do_birth
            @birth(Sample, idx)
            @write(new_tr[:kernel => :num_samples], current_num_samples + 1, :disc)

            new_val = @read(fwd_prop_tr[:new_val], :cont)
            @write(new_tr[:world => :val => Sample(idx) => :val], new_val, :cont)
        else
            @death(Sample, idx)
            @write(new_tr[:kernel => :num_samples], current_num_samples - 1, :disc)
            
            current_val = @read(old_tr[:world => :val => Sample(idx)], :cont)
            @write(bwd_prop_tr[:new_val], current_val, :cont)
        end
        @write(bwd_prop_tr[:do_birth], !do_birth, :disc)
        @write(bwd_prop_tr[:idx], idx, :disc)
    end
    birth_death_mh_kern = OUPMMHKernel(birth_death_proposal, (), bd_involution)
    new_tr = run_mh_100(tr, birth_death_mh_kern, obs)
    @test get_score(new_tr) > get_score(tr)
end