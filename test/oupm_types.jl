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
end