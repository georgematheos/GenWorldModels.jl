@type AudioSource

@gen (static) function get_volume(world::World, source::AudioSource)
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
    :volume => get_volume
)

@testset "basic OUPM type usage in UsingWorld" begin
@testset "simple lookup_or_generate conversion - no origins" begin
    @gen function _simple_convert(world)
        src ~ lookup_or_generate(world[:abstract][AudioSource(1)])
        idx ~ lookup_or_generate(world[:index][src])
        return (idx, src)
    end
    simple_convert = UsingWorld(_simple_convert)

    tr = simulate(simple_convert, ())
    @test get_retval(tr)[1] == 1
    @test get_retval(tr)[2] isa AbstractOUPMObject{:AudioSource}
end
@testset "automatic conversion at lookup_or_generate for generate - no origins" begin
    tr = simulate(get_vols_and_inds, ())
    inds_to_vols = get_retval(tr)
    for ((out_index, _), in_index) in zip(inds_to_vols, tr[:kernel => :samples])
        @test out_index == in_index
    end
end

@testset "simple lookup_or_generate conversion -- with origins" begin
    @gen function _origin_convert(world)
        t1 ~ lookup_or_generate(world[:abstract][Timestep(1)])
        t2 ~ lookup_or_generate(world[:abstract][Timestep(2)])
        blip ~ lookup_or_generate(world[:abstract][Blip((t1, t2), 1)])
        idx ~ lookup_or_generate(world[:index][blip])
        origin ~ lookup_or_generate(world[:origin][blip])
        concrete1 ~ lookup_or_generate(world[:concrete][origin[1]])
        concrete2 ~ lookup_or_generate(world[:concrete][origin[2]])
        return ((concrete1, concrete2, idx), (t1, t2, blip), origin)
    end
    origin_convert = UsingWorld(_origin_convert)

    tr = simulate(origin_convert, ())
    @test get_retval(tr)[1] == (Timestep(1), Timestep(2), 1)
    @test get_retval(tr)[2] isa Tuple{AbstractOUPMObject{:Timestep}, AbstractOUPMObject{:Timestep}, AbstractOUPMObject{:Blip}}
    @test get_retval(tr)[3] isa Tuple{AbstractOUPMObject{:Timestep}, AbstractOUPMObject{:Timestep}}
    @test get_score(tr) == 0.
end

@testset "simple lookup_or_generate conversion -- including auto origin -> abstract" begin
    @gen function _origin_convert(world)
        t3 ~ lookup_or_generate(world[:abstract][Timestep(3)])
        blip ~ lookup_or_generate(world[:abstract][Blip((Timestep(1), Timestep(2), t3), 1)])
        idx ~ lookup_or_generate(world[:index][blip])
        origin ~ lookup_or_generate(world[:origin][blip])
        concrete1 ~ lookup_or_generate(world[:concrete][origin[1]])
        concrete2 ~ lookup_or_generate(world[:concrete][origin[2]])
        return ((concrete1, concrete2, idx), blip, origin, t3)
    end
    origin_convert = UsingWorld(_origin_convert)
    tr = simulate(origin_convert, ())
    @test get_retval(tr)[1] == (Timestep(1), Timestep(2), 1)
    @test get_retval(tr)[2] isa AbstractOUPMObject{:Blip}
    @test get_retval(tr)[3] isa Tuple{AbstractOUPMObject{:Timestep}, AbstractOUPMObject{:Timestep}, AbstractOUPMObject{:Timestep}}
    @test get_retval(tr)[3][3] == get_retval(tr)[4]
    @test get_score(tr) == 0.
end

end