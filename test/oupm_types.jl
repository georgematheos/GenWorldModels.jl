@type AudioSource

@gen (static) function get_volume(world::World, source::AudioSource)
    vol ~ uniform_discrete(20, 100)
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

@testset "open universe types - basics" begin
    @test AudioSource(1) isa AudioSource{Int}
    @test AudioSource(uuid4()) isa AudioSource{UUID}
    @test_throws MethodError AudioSource("i am a string")

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

# @testset "open universe types - `UsingWorld` functionality"
#     get_vols_and_inds = UsingWorld(
#         _vol_ind_kernel,
#         :volume => get_volume;
#         oupm_types=(AudioSource,)
#     )
#     tr = simulate(get_vols_and_inds, ())
#     idx = tr[:kernel => :samples][1]
#     source = AudioSource(idx)
    
#     # TODO: update tests below here

#     @test !isempty(get_subtree(get_choices(tr), :world => :volume => source))
    
#     acc = false
#     new_tr = nothing
#     while !acc
#         new_tr, acc = mh(tr, select(:world => :volume => source))
#     end
#     @test new_tr[:world => :volume => source] !== tr[:world => :volume => source]
#     new_inds_to_vols = get_retval(new_tr)
#     old_inds_to_vols = get_retval(old_tr)

#     for (old, new) in zip(old_inds_to_vols, new_inds_to_vols)
#         old_idx, old_vol = old
#         new_idx, new_vol = new
#         @test old_idx == new_idx
#         if old_idx == idx
#             @test old_vol != new_vol
#         else
#             @test old_vol == new_vol
#         end
#     end

#     # should throw error if we don't specify that AudioSource is an OUPMType
#     broken_get_vols_and_inds = UsingWorld(_vol_ind_kernel, :volume => get_volume)
#     @test_throws Exception simulate(broken_get_vols_and_inds, ())

#     # TODO: test updating with diffs enabled
# end