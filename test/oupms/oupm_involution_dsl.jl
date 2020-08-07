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
observe_sample_sum = UsingWorld(observe_samples_sum_kernel, :val => get_val)

@testset "OUPM move involution DSL" begin
    OBS = 3.
    tr, _  = generate(observe_sample_sum, (), choicemap(
        (:kernel => :num_samples, 4), (:kernel => :observation, OBS),
    ))

    function run_mh_20(tr, kern, obs)
        new_tr = tr
        for i=1:20
            new_tr, acc = mh(new_tr, kern; check=true, observations=obs)
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
    @oupm_involution invalid_bd_involution (old_tr, fwd_prop_tr) to (new_tr, bwd_prop_tr) begin
        idx = @read(fwd_prop_tr[:idx], :disc)
        current_num_samples = @read(old_tr[:kernel => :num_samples], :disc)
        if @read(fwd_prop_tr[:do_birth], :disc)
            @birth(Sample(idx))
            @write(new_tr[:kernel => :num_samples], current_num_samples + 1, :disc)
            new_val = @read(fwd_prop_tr[:new_val], :cont)
            @write(new_tr[:world => :val => Sample(idx) => :val], new_val, :cont)
            @write(bwd_prop_tr[:do_birth], false, :disc)
        else
            @death(Sample(idx))
            @write(new_tr[:kernel => :num_samples], current_num_samples - 1, :disc)
            @write(bwd_prop_tr[:do_birth], true, :disc)
        end
        @write(bwd_prop_tr[:idx], idx, :disc)
    end

    invalid_birth_death_mh_kern = OUPMMHKernel(birth_death_proposal, (), invalid_bd_involution)
    @test_logs (:error, ) match_mode=:any (@test_throws Exception run_mh_20(tr, invalid_birth_death_mh_kern, obs))

    # now check a valid one!
    @oupm_involution bd_involution (old_tr, fwd_prop_tr) to (new_tr, bwd_prop_tr) begin
        idx = @read(fwd_prop_tr[:idx], :disc)
        do_birth = @read(fwd_prop_tr[:do_birth], :disc)
        current_num_samples = @read(old_tr[:kernel => :num_samples], :disc)
        if do_birth
            @birth(Sample(idx))
            @write(new_tr[:kernel => :num_samples], current_num_samples + 1, :disc)

            new_val = @read(fwd_prop_tr[:new_val], :cont)
            @write(new_tr[:world => :val => Sample(idx) => :val], new_val, :cont)
        else
            @death(Sample(idx))
            @write(new_tr[:kernel => :num_samples], current_num_samples - 1, :disc)
            
            # could use copy here, but want to test continuous read/write behavior
            current_val = @read(old_tr[:world => :val => Sample(idx) => :val], :cont)
            @write(bwd_prop_tr[:new_val], current_val, :cont)
        end
        @write(bwd_prop_tr[:do_birth], !do_birth, :disc)
        @write(bwd_prop_tr[:idx], idx, :disc)
    end
    birth_death_mh_kern = OUPMMHKernel(birth_death_proposal, (), bd_involution)
    new_tr = run_mh_20(tr, birth_death_mh_kern, obs)

    ### Split/Merge ###
    @gen function split_merge_proposal(tr)
        current_num_samples = tr[:kernel => :num_samples]
        do_split ~ bernoulli(0.5)
        if do_split
            solo_idx ~ uniform_discrete(1, current_num_samples)
            deuce_idx1 ~ uniform_discrete(1, current_num_samples + 1)
            deuce_idx2 ~ uniform_discrete(1, current_num_samples + 1)
            if deuce_idx1 != deuce_idx2
                old_val = tr[:world => :val => Sample(solo_idx) => :val]
                new_val1 ~ normal(old_val, 0.5)
                new_val2 ~ normal(old_val, 0.5)
            end
        else
            solo_idx ~ uniform_discrete(1, current_num_samples - 1)
            deuce_idx1 ~ uniform_discrete(1, current_num_samples)
            deuce_idx2 ~ uniform_discrete(1, current_num_samples)
            if deuce_idx1 != deuce_idx2
                old_val1 = tr[:world => :val => Sample(deuce_idx1) => :val]
                old_val2 = tr[:world => :val => Sample(deuce_idx2) => :val]
                new_val ~ normal(old_val1 + old_val2, 1.)
            end
        end
    end
    @oupm_involution split_merge_involution (old_tr, fwd_prop_tr) to (new_tr, bwd_prop_tr) begin
        deuce_idx1 = @read(fwd_prop_tr[:deuce_idx1], :disc)
        deuce_idx2 = @read(fwd_prop_tr[:deuce_idx2], :disc)
        do_split = @read(fwd_prop_tr[:do_split], :disc)
        solo_idx = @read(fwd_prop_tr[:solo_idx], :disc)
        if deuce_idx1 != deuce_idx2
            current_num_samples = @read(old_tr[:kernel => :num_samples], :disc)
            if do_split
                @split(Sample(solo_idx), deuce_idx1, deuce_idx2)
                @write(new_tr[:kernel => :num_samples], current_num_samples + 1, :disc)
                @copy(fwd_prop_tr[:new_val1], new_tr[:world => :val => Sample(deuce_idx1) => :val])
                @copy(fwd_prop_tr[:new_val2], new_tr[:world => :val => Sample(deuce_idx2) => :val])
                @copy(old_tr[:world => :val => Sample(solo_idx) => :val], bwd_prop_tr[:new_val])
            else
                @merge(Sample(solo_idx), deuce_idx1, deuce_idx2, ())
                @write(new_tr[:kernel => :num_samples], current_num_samples - 1, :disc)
                @copy(fwd_prop_tr[:new_val], new_tr[:world => :val => Sample(solo_idx) => :val])
                @copy(old_tr[:world => :val => Sample(deuce_idx1) => :val], bwd_prop_tr[:new_val1])
                @copy(old_tr[:world => :val => Sample(deuce_idx2) => :val], bwd_prop_tr[:new_val2])
            end
        end
        @copy(fwd_prop_tr[:deuce_idx1], bwd_prop_tr[:deuce_idx1])
        @copy(fwd_prop_tr[:deuce_idx2], bwd_prop_tr[:deuce_idx2])
        @write(bwd_prop_tr[:do_split], !do_split, :disc)
        @copy(fwd_prop_tr[:solo_idx], bwd_prop_tr[:solo_idx])
    end
    split_merge_mh_kern = OUPMMHKernel(split_merge_proposal, (), split_merge_involution)
    new_tr = run_mh_20(tr, split_merge_mh_kern, obs)

    ### Moving stuff ###
    @gen function move_proposal(tr)
        num_samples = tr[:kernel => :num_samples]
        from_idx ~ uniform_discrete(1, num_samples)
        to_idx ~ uniform_discrete(1, num_samples)
    end
    @oupm_involution move_inv (old_tr, fwd_prop_tr) to (new_tr, bwd_prop_tr) begin
        from = @read(fwd_prop_tr[:from_idx], :disc)
        to = @read(fwd_prop_tr[:to_idx], :disc)
        @move(Sample(from), Sample(to))
        @copy(fwd_prop_tr[:from_idx], bwd_prop_tr[:to_idx])
        @copy(fwd_prop_tr[:to_idx], bwd_prop_tr[:from_idx])
    end
    move_mh_kern = OUPMMHKernel(move_proposal, (), move_inv)
    new_tr = run_mh_20(tr, move_mh_kern, obs)

    ### Regenerating ###
    @gen function bd_prop_regen(model_tr)
        do_birth ~ bernoulli(0.5)
        current_num_samples = model_tr[:kernel => :num_samples]
        if do_birth
            idx ~ uniform_discrete(1, current_num_samples + 1)
        else
            idx ~ uniform_discrete(1, current_num_samples)
        end
    end
    @oupm_involution bd_inv_regen (old_tr, fwd_prop_tr) to (new_tr, bwd_prop_tr) begin
        idx = @read(fwd_prop_tr[:idx], :disc)
        do_birth = @read(fwd_prop_tr[:do_birth], :disc)
        current_num_samples = @read(old_tr[:kernel => :num_samples], :disc)
        if do_birth
            @birth(Sample(idx))
            @write(new_tr[:kernel => :num_samples], current_num_samples + 1, :disc)
            @regenerate(:world => :val => Sample(idx))
        else
            @death(Sample(idx))
            @write(new_tr[:kernel => :num_samples], current_num_samples - 1, :disc)
            @save_for_reverse_regenerate(:world => :val => Sample(idx))
        end
        @write(bwd_prop_tr[:do_birth], !do_birth, :disc)
        @write(bwd_prop_tr[:idx], idx, :disc)
    end
    bd_regen_mh_kern = OUPMMHKernel(bd_prop_regen, (), bd_inv_regen)
    new_tr = run_mh_20(tr, bd_regen_mh_kern, obs)

    # now do some simple checks on the acceptance ratio
    prop_tr, _ = generate(bd_prop_regen, (tr,), choicemap((:do_birth, true)))
    (new_tr, weight, bwd_prop_tr, log_abs_det) = GenWorldModels.symmetric_trace_translator_run_transform(bd_inv_regen, tr, prop_tr, bd_prop_regen, ())
    num = tr[:kernel => :num_samples]
    original_sum = sum(tr[:kernel => :samples])
    new_val = new_tr[:world => :val => Sample(prop_tr[:idx])]
    println("original sum: $original_sum | original num: $num | new val: $new_val")
    num_change_weight = logpdf(poisson, num+1, 5) - logpdf(poisson, num, 5)
    obsprob_change_weight = logpdf(normal, original_sum + new_val, OBS, 1.) - logpdf(normal, original_sum, OBS, 1.)
    expected_weight = num_change_weight + obsprob_change_weight
    @test isapprox(weight, expected_weight)
    @test log_abs_det == 0.

    (new_tr, weight, bwd_trace, log_abs_det) = GenWorldModels.symmetric_trace_translator_run_transform(bd_inv_regen, new_tr, bwd_prop_tr, bd_prop_regen, ())
    @test isapprox(weight, -expected_weight)
    @test log_abs_det == 0.
end

@testset "num change without OUPM move" begin
    tr, _ = generate(observe_sample_sum, (), choicemap(
        (:kernel => :num_samples, 2),
        (:world => :val => Sample(1) => :val, 0.4 ))
    )
    OBS = tr[:kernel => :observation]
    for _=1:5
        new_tr, weight, _ = regenerate(tr, (), (), select(:kernel => :num_samples))
        new_sum = sum(new_tr[:kernel => :samples])
        old_sum = sum(tr[:kernel => :samples])
        obsscore_diff = logpdf(normal, new_sum, OBS, 1.) - logpdf(normal, old_sum, OBS, 1.)
        @test isapprox(weight, obsscore_diff)
        tr = new_tr
    end
end

# TODO: test involution DSL in a case where there are origin moves for split/merge moves?