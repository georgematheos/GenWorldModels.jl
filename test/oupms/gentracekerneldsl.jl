@type Sample
@dist poisson_plus_1(l) = poisson(l) + 1
@gen (static) function get_val(world::World, sample::Sample)
    val ~ normal(1, 0.5)
    return val
end
@gen (static) function observe_samples_sum_kernel(world::World)
    num_samples ~ poisson_plus_1(5)
    samples ~ Map(lookup_or_generate)([world[:val][Sample(i)] for i=1:num_samples])
    total = reduce(+, samples, init=0.)
    observation ~ normal(total, 1)
    return observation
end
@load_generated_functions()
observe_sample_sum = UsingWorld(observe_samples_sum_kernel, :val => get_val)

@testset "OUPM Kernel DSL" begin
    OBS = 3.
    tr, _  = generate(observe_sample_sum, (), choicemap(
        (:kernel => :num_samples, 4), (:kernel => :observation, OBS),
    ))

    function run_mh_20(tr, kern, obs; kwargs...)
        new_tr = tr
        for i=1:20
            new_tr, acc = mh(new_tr, kern; check=true, observations=obs, kwargs...)
        end
        new_tr
    end
    obs = choicemap((:kernel => :observation, OBS))

    @kernel function invalid_birth_death_kernel(trace)
        do_birth ~ bernoulli(0.5)
        current_num_samples = trace[:kernel => :num_samples]
        if do_birth
            idx ~ uniform_discrete(1, current_num_samples + 1)
            current_total = sum(tr[:world => :val => Sample(i)] for i=1:current_num_samples)
            expected_val = tr[:kernel => :observation] - current_total
            new_val ~ normal(expected_val, 2.)
            
            update = WorldUpdate(
                Create(Sample(idx)),
                choicemap(
                    (:kernel => :num_samples, current_num_samples + 1),
                    (:world => :val => Sample(idx) => :val, new_val),
                )
            )
        else
            idx ~ uniform_discrete(1, current_num_samples)
            update = WorldUpdate(
                Delete(Sample(idx)),
                choicemap((:kernel => :num_samples, current_num_samples - 1))
            )
        end

        # invalid reverse move
        return (update, choicemap())
    end
    # invalid_birth_death_mh_kern = MHProposal(invalid_birth_death_kernel)
    # @test_logs (:error, ) match_mode=:any (@test_throws Exception run_mh_20(tr, invalid_birth_death_mh_kern, obs))

    ### Birth-Death ###
    @kernel function birth_death_kernel(trace)
        do_birth ~ bernoulli(0.5)
        current_num_samples = trace[:kernel => :num_samples]
        if do_birth
            idx ~ uniform_discrete(1, current_num_samples + 1)
            current_total = sum(trace[:world => :val => Sample(i)] for i=1:current_num_samples)
            expected_val = trace[:kernel => :observation] - current_total
            new_val ~ normal(expected_val, 2.)
            
            return (
                WorldUpdate(
                    Create(Sample(idx)),
                    choicemap(
                        (:kernel => :num_samples, current_num_samples + 1),
                        (:world => :val => Sample(idx) => :val, new_val),
                    )
                ),
                choicemap((:do_birth, !do_birth), (:idx, idx))
            )
        else
            idx ~ uniform_discrete(1, current_num_samples)
            return (
                WorldUpdate(
                    Delete(Sample(idx)),
                    choicemap((:kernel => :num_samples, current_num_samples - 1))
                ),
                choicemap((:do_birth, !do_birth), (:idx, idx), (:new_val, trace[:world => :val => Sample(idx) => :val]))
            )
        end
    end

    birth_death_mh_kern = MHProposal(birth_death_kernel)
    new_tr = run_mh_20(tr, birth_death_mh_kern, obs)

    ### Split/Merge ###
    @kernel function split_merge_kernel(tr)
        current_num_samples = tr[:kernel => :num_samples]
        do_split ~ bernoulli(0.5)
        if do_split
            solo_idx ~ uniform_discrete(1, current_num_samples)
            deuce_idx1 ~ uniform_discrete(1, current_num_samples + 1)
            deuce_idx2 ~ uniform_discrete(1, current_num_samples + 1)

            if deuce_idx1 == deuce_idx2
                update = EmptyAddressTree()
                bwd = choicemap()
            else
                old_val = tr[:world => :val => Sample(solo_idx) => :val]
                new_val1 ~ normal(old_val, 0.5)
                new_val2 ~ normal(old_val, 0.5)

                update = WorldUpdate(
                    Split(Sample(solo_idx), deuce_idx1, deuce_idx2),
                    choicemap(
                        (:kernel => :num_samples, current_num_samples + 1),
                        (:world => :val => Sample(deuce_idx1) => :val, new_val1),
                        (:world => :val => Sample(deuce_idx2) => :val, new_val2)    
                    )
                )
                bwd = choicemap(
                    (:new_val, tr[:world => :val => Sample(solo_idx) => :val])
                )
            end
        else
            solo_idx ~ uniform_discrete(1, current_num_samples - 1)
            deuce_idx1 ~ uniform_discrete(1, current_num_samples)
            deuce_idx2 ~ uniform_discrete(1, current_num_samples)
            if deuce_idx1 == deuce_idx2
                update = EmptyAddressTree()
                bwd = choicemap()
            else
                old_val1 = tr[:world => :val => Sample(deuce_idx1) => :val]
                old_val2 = tr[:world => :val => Sample(deuce_idx2) => :val]
                new_val ~ normal(old_val1 + old_val2, 1.)

                update = WorldUpdate(
                    Merge(Sample(solo_idx), deuce_idx1, deuce_idx2),
                    choicemap(
                        (:kernel => :num_samples, current_num_samples - 1),
                        (:world => :val => Sample(solo_idx) => :val, new_val)
                    )
                )
                bwd = choicemap(
                    (:new_val1, tr[:world => :val => Sample(deuce_idx1) => :val]),
                    (:new_val2, tr[:world => :val => Sample(deuce_idx2) => :val])
                )
            end
        end
        bwd[:deuce_idx1] = deuce_idx1
        bwd[:deuce_idx2] = deuce_idx2
        bwd[:solo_idx] = solo_idx
        bwd[:do_split] = !do_split
        
        return (update, bwd)
    end
    split_merge_mh_kern = MHProposal(split_merge_kernel)
    new_tr = run_mh_20(tr, split_merge_mh_kern, obs)

    ### Move stuff ###
    @kernel function move_kernel(tr)
        num_samples = tr[:kernel => :num_samples]
        from_idx ~ uniform_discrete(1, num_samples)
        to_idx ~ uniform_discrete(1, num_samples)
        return (
            WorldUpdate(Move(Sample(from_idx), Sample(to_idx))),
            choicemap(
                (:from_idx, to_idx),
                (:to_idx, from_idx)
            )    
        )
    end

    move_mh_kern = MHProposal(move_kernel)
    new_tr = run_mh_20(tr, move_mh_kern, obs)

    ### regeneration ###
    @kernel function birth_death_regen_kernel(tr)
        do_birth ~ bernoulli(0.5)
        current_num_samples = tr[:kernel => :num_samples]
        if do_birth
            idx ~ uniform_discrete(1, current_num_samples + 1)

            return (
                WorldUpdate( # update spec
                    Create(Sample(idx)),
                    regenchoicemap(
                        (:kernel => :num_samples, current_num_samples + 1),
                        (:world => :val => Sample(idx), AllSelection())
                    )
                ),
                choicemap((:do_birth, false), (:idx, idx)) # bwd choices
            )
        else
            idx ~ uniform_discrete(1, current_num_samples)
            return (
                WorldUpdate( # update spec
                    Delete(Sample(idx)),
                    choicemap(
                        (:kernel => :num_samples, current_num_samples - 1)
                    )
                ),
                (
                    choicemap((:do_birth, true), (:idx, idx)), # bwd choices
                    select(:world => :val => Sample(idx)) # addresses regenerated in the reverse move
                )
            )
        end
    end

    bd_regen_mh_kern = MHProposal(birth_death_regen_kernel)
    new_tr = run_mh_20(tr, bd_regen_mh_kern, obs)

    # now do some simple checks on the acceptance ratio
    # error("Acceptance ratio tests not yet implemented.")
    # TODO!!
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