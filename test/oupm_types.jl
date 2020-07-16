struct AudioSource <: OUPMType
    id::UUID
end

@gen (static) function get_volume(world::World, source::AudioSource)
    vol ~ uniform_discrete(20, 100)
    return vol
end

@gen (static) function _vol_ind_kernel(world::World)
    num_sources ~ uniform_discrete(3, 7)
    num_samples ~ uniform_discrete(1, 3)
    samples ~ Map(uniform_discrete)(fill(1, num_samples), fill(num_sources, num_samples))
    sources = [AudioSource(world, idx) for idx in samples]
    volumes ~ Map(lookup_or_generate)([world[:volume][source] for source in sources])
    indices ~ Map(lookup_or_generate)([world[:index][source] for source in sources])
    return (indices, volumes)
end

@load_generated_functions()

@testset "open universe types - basics" begin
    get_vols_and_inds = UsingWorld(
        _vol_ind_kernel,
        :volume => get_volume;
        oupm_types=(AudioSource,)
    )
    tr = simulate(get_vols_and_inds, ())

    # should throw error if we don't specify that AudioSource is an OUPMType
    broken_get_vols_and_inds = UsingWorld(_vol_ind_kernel, :volume => get_volume)
    @test_throws Exception simulate(broken_get_vols_and_inds, ())
end