@type AudioSource

@gen (static) function get_volume(world::World, source::AudioSource{UUID})
    vol ~ normal(100, 4)
    idx ~ lookup_or_generate(world[:index][source])
    return idx => vol
end

@gen (static) function _vol_ind_kernel(world::World)
    num_sources ~ uniform_discrete(3, 7)
    num_samples ~ uniform_discrete(2, 3)
    samples ~ Map(uniform_discrete)(fill(1, num_samples), fill(num_sources, num_samples))
    sources = [AudioSource(idx) for idx in samples]
    idx_vol_pairs ~ Map(lookup_or_generate)([world[:volume][source] for source in sources])
    return idx_vol_pairs
end

@load_generated_functions()

get_vols_and_inds = UsingWorld(
    _vol_ind_kernel,
    :volume => get_volume;
    oupm_types=(AudioSource,)
)

@testset "open universe types" begin
@testset "type construction" begin
    @test AudioSource(1) isa AudioSource{Int}
    @test AudioSource(uuid4()) isa AudioSource{UUID}
    @test_throws MethodError AudioSource("i am a string")
    @test GenWorldModels.oupm_type(AudioSource(1)) == AudioSource
    @test GenWorldModels.oupm_type(AudioSource(uuid4())) == AudioSource
end
@testset "simple lookup_or_generate conversion" begin
    @gen function _simple_convert(world)
        src ~ lookup_or_generate(world[AudioSource][1])
        idx ~ lookup_or_generate(world[:index][src])
        return (idx, src)
    end
    simple_convert = UsingWorld(_simple_convert; oupm_types=(AudioSource,))

    tr = simulate(simple_convert, ())
    @test get_retval(tr)[1] == 1
    @test get_retval(tr)[2] isa AudioSource{UUID}
end
@testset "automatic conversion at lookup_or_generate for generate" begin
    tr = simulate(get_vols_and_inds, ())
    inds_to_vols = get_retval(tr)
    for ((out_index, _), in_index) in zip(inds_to_vols, tr[:kernel => :samples])
        @test out_index == in_index
    end
end

@testset "simple regenerate in `UsingWorld` with idx object in address" begin
    tr = simulate(get_vols_and_inds, ())
    idx = tr[:kernel => :samples][1]
    source = AudioSource(idx)
    
    @test !isempty(get_subtree(get_choices(tr), :world => :volume => source))
    
    acc = false
    new_tr = nothing
    while !acc # shouldn't need this; this should be a gibbs update.
        new_tr, acc = mh(tr, select(:world => :volume => source))
    end
    @test new_tr[:world => :volume => source] !== tr[:world => :volume => source]
    new_inds_to_vols = get_retval(new_tr)
    old_inds_to_vols = get_retval(tr)

    for (old, new) in zip(old_inds_to_vols, new_inds_to_vols)
        old_idx, old_vol = old
        new_idx, new_vol = new
        @test old_idx == new_idx
        if old_idx == idx
            @test old_vol != new_vol
        else
            @test old_vol == new_vol
        end
    end

    @test new_tr[:world => :volume => source] == (idx => get_choices(new_tr)[:world => :volume => source => :vol])

    # examine the internal `regenerate` call which should have occurred
    new_tr, weight, retdiff, discard = update(tr, get_args(tr), Tuple(NoChange() for arg in get_args(tr)), select(:world => :volume => source), EmptySelection())
    @test discard == choicemap((:world => :volume => source => :vol, tr[:world => :volume => source => :vol]))

    # P[new_tr]/P[old_tr] * Q[old_tr|selection]/Q[new_tr|selection] = 0 since here Q and P are the same since there are no downstream choices
    @test weight == 0.

    # should throw error if we don't specify that AudioSource is an OUPMType
    broken_get_vols_and_inds = UsingWorld(_vol_ind_kernel, :volume => get_volume)
    @test_throws Exception simulate(broken_get_vols_and_inds, ())

    # TODO: test updating with diffs enabled
end

@testset "oupm moves - basics" begin
    constraints = choicemap(
        (:kernel => :num_sources, 5),
        (:kernel => :num_samples, 2),
        (:kernel => :samples => 1, 2),
        (:kernel => :samples => 2, 3)
    )
    tr, weight = generate(get_vols_and_inds, (), constraints)
    @test isapprox(weight, -(log(5) + log(2) + 2*log(5)))
    ch = get_choices(tr)

    ### BIRTH ###

    # birth a new object at index 3.  Since we are looking up index 3 (but not index 4), this means the new index 3 will be
    # generated newly, and we discard the old index 3 (which is now at index 4 and is not looked up).
    spec = UpdateWithOUPMMovesSpec((BirthMove(AudioSource, 3),), choicemap((:kernel => :num_sources, 6)))
    new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
    newch = get_choices(new_tr)
    s2 = AudioSource(2); s3 = AudioSource(3)
    @test get_subtree(newch, :world => :volume => s2) == get_subtree(ch, :world => :volume => s2)
    @test length(collect(get_subtrees_shallow(get_subtree(newch, :world => :volume)))) == 2
    @test new_tr[:world => :volume => s3] != tr[:world => :volume => s3] # should have been regenerated
    # generating weight not included since this is cancelled out by the Q distribution
    expected_weight = -logpdf(normal, tr[:world => :volume => s3 => :vol], 100, 4) + 2*(log(1/6) - log(1/5))
    @test isapprox(expected_weight, weight)
    @test new_tr[][1] == tr[][1]
    @test new_tr[][2] != tr[][2]

    @test reverse_move isa UpdateWithOUPMMovesSpec
    @test reverse_move.moves == (DeathMove(AudioSource, 3),)
    expected_reverse_constraints = choicemap((:kernel => :num_sources, 5))
    set_submap!(expected_reverse_constraints, :world => :volume => s3, get_submap(ch, :world => :volume => s3))
    @test reverse_move.subspec == expected_reverse_constraints

    # birth a new object at index 2.  Should generate new trace at index 2, move current trace at 2 to 3, and discard current 3.
    spec = UpdateWithOUPMMovesSpec((BirthMove(AudioSource, 2),), choicemap((:kernel => :num_sources, 6)))
    new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
    newch = get_choices(new_tr)
    @test get_subtree(newch, :world => :volume => s3) == get_subtree(ch, :world => :volume => s2)
    @test length(collect(get_subtrees_shallow(get_subtree(newch, :world => :volume)))) == 2
    @test new_tr[:world => :volume => s2] != tr[:world => :volume => s2] # should have been regenerated
    expected_weight = -logpdf(normal, tr[:world => :volume => s3 => :vol], 100, 4) + 2*(log(1/6) - log(1/5))
    @test isapprox(expected_weight, weight)
    @test new_tr[][2].second == tr[][1].second
    @test new_tr[][1] != tr[][1]

    @test reverse_move isa UpdateWithOUPMMovesSpec
    @test reverse_move.moves == (DeathMove(AudioSource, 2),)
    expected_reverse_constraints = choicemap((:kernel => :num_sources, 5))
    set_submap!(expected_reverse_constraints, :world => :volume => s3, get_submap(ch, :world => :volume => s3))
    @test reverse_move.subspec == expected_reverse_constraints
    # display(discard)
    # println("exp:")
    # display(expected_discard)


end

end