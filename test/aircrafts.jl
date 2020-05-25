# aircraft model from the `readme`

module Aircrafts
using Gen

include("../src/WorldModels.jl")
using .WorldModels

@gen function position_prior()
    return 0
end

@gen function aircraft_movement_model(prev_position)
    delta_x ~ exponential(4)
    return prev_position + delta_x
end

@gen function take_measurement(world, arg)
    (aircraft_index, time) = arg
    position ~ lookup_or_generate(world[:positions][(aircraft_index, time)])
    measured ~ normal(position, 0.5)
    return measured
end

@gen function generate_position(world, arg)
    (aircraft_index, time) = arg
    if time == 1
        position ~ position_prior()
    else
        prev_position ~ lookup_or_generate(world[:positions][(aircraft_index, time - 1)])
        position ~ aircraft_movement_model(prev_position)
    end
    return position
end

@gen function measure_aircraft_at_time(world, aircraft_index, time)
    measurement ~ lookup_or_generate(world[:measurements][(aircraft_index, time)])
    return measurement
end
measure_aircrafts_at_times = Map(measure_aircraft_at_time)

@gen function kernel(world, time_to_measure_at)
    num_aircrafts ~ poisson(5)
    measurements ~ measure_aircrafts_at_times(fill(world, num_aircrafts), collect(1:num_aircrafts), fill(time_to_measure_at, num_aircrafts))
    return measurements
end

get_measurements_at_time = UsingWorld(kernel, :measurements => take_measurement, :positions => generate_position)

# example usage:
measurements_at_time_5 = get_measurements_at_time(5)

tr, _ = generate(get_measurements_at_time, (3,), choicemap((:kernel => :num_aircrafts, 3)))
end