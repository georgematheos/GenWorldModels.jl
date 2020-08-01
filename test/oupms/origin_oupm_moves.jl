@dist num_aircrafts(_::World, _::Tuple{}) = poisson(5)
@dist num_blips(_::World, _::Tuple{<:Aircraft, <:Timestep}) = poisson(1)
@dist plane_size(_::World, _::Aircraft) = normal(600, 100)
@gen (static) function blip_size(world, b::Blip)
    (aircraft, timestep) = {:origin} ~ lookup_or_generate(world[:origin][b])
    plane_size ~ lookup_or_generate(world[:plane_size][aircraft])
    blip_size ~ normal(plane_size/600, 0.05)
    return blip_size
end

# returns a dict of (aircraft_idx, blip_idx_for_time_and_aircraft) => blip_size for every blip at this time
@gen (static) function blip_sizes_at_time(world, time)
    n_a ~ lookup_or_generate(world[:num_aircrafts][()])
    blips_per_aircraft ~ Map(lookup_or_generate)([world[:num_blips][(Aircraft(i), time)] for i=1:n_a])
    pairs = [(i, j) for i=1:n_a for j=1:blips_per_aircraft[i]]
    blips = [Blip((Aircraft(i), time), j) for (i, j) in pairs]
    sizes ~ Map(lookup_or_generate)([world[:blip_size][blip] for blip in blips])
    return Dict([pair => size for (pair, size) in zip(pairs, sizes)])
end

@gen (static) function _get_blip_sizes_at_times(world, timesteps)
    blips_sizes_per_time ~ Map(blip_sizes_at_time)(fill(world, length(timesteps)), [Timestep(i) for i in timesteps])
    return Dict([
        Blip((Aircraft(i), Timestep(timestep)), j) => size
        for (timestep, dict) in zip(timesteps, blips_sizes_per_time)
        for ((i, j), size) in dict
    ])
end
get_blip_sizes_at_times = UsingWorld(_get_blip_sizes_at_times, :num_aircrafts => num_aircrafts, :num_blips => num_blips, :plane_size => plane_size, :blip_size => blip_size)
@load_generated_functions()

@testset "OUPM moves for origin objects" begin
    tr, _ = generate(get_blip_sizes_at_times, ([1, 2, 3],), choicemap(
        (:world => :num_aircrafts => (), 3),
        (:world => :num_blips => (Aircraft(1), Timestep(1)), 1),
        (:world => :num_blips => (Aircraft(2), Timestep(1)), 1),
        (:world => :num_blips => (Aircraft(3), Timestep(1)), 1),
        (:world => :num_blips => (Aircraft(1), Timestep(2)), 1),
        (:world => :num_blips => (Aircraft(2), Timestep(2)), 1),
        (:world => :num_blips => (Aircraft(3), Timestep(2)), 1)
    ))
    @test get_retval(tr) isa Dict{ConcreteIndexOUPMObject{:Blip, Tuple{ConcreteIndexOUPMObject{:Aircraft, Tuple{}}, ConcreteIndexOUPMObject{:Timestep, Tuple{}}}}, Float64}
end