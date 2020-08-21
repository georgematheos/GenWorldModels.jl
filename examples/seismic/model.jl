#=
Questions:
- 4: To clarify, here, σ^2 is drawn from InvGamma.  Then, in 12, we use σ as a parameter--which means
we need to take the sqrt of the draw from InvGamma.  Is this correct, or should we draw σ from InvGamma?
- 14: should the number of noise detections be Poisson(F(s)) (mean=false alarm rate)
or Poisson(N(s)) (mean=noise level)
=#

@type Station

#################
# Physics model #
#################

@dist event_occurance_rate(_, _) = gamma(α_I, β_I)
@dist background_noise_level(_, _::Station) = inv_gamma(α_N, β_N)
@dist false_alarm_rate(_, _::Station) = gamma(α_F, β_F)
@gen (static) function signal_velocity(w, o)
    W_0 ~ normal(μ_V, σ2_V)
    return W_0^2
end
@gen (static) function absorpivity_per_unit_distance(w, o)
    β_0 ~ normal(μ_B, σ2_B)
    return β_0^2
end
@dist ν(_, _::Station) = normal(μ_ν, σ2_ν)
@dist σ2(_, _::Station) = inv_gamma(α_S, β_S)
@dist arrival_time_measurement_error_params(_, _::Station) = normal_inverse_gamma(μ_t, λ_t, α_t, β_t)
@dist amplitude_measurement_error_params(_, _::Station) = normal_inverse_gamma(μ_a, λ_a, α_a, β_a)
@dist noise_detection_amplitude_params(_, _::Station) = normal_inverse_gamma(μ_n, λ_n, α_n, β_n)

@gen (static) function station_location(world, s::Station)
    idx ~ lookup_or_generate(world[:index][s])
    return (idx - 1) * 0.25
end

@gen (static) function travel_time(world, x, y)
    vel ~ lookup_or_generate(world[:signal_velocity][()])
    return abs(x-y)/vel
end

###############
# Event model #
###############

@type Event

@gen (static) function num_events(world, o)
    rate ~ lookup_or_generate(world[:event_occurance_rate][()])
    num ~ poisson(rate)
    return num
end

@dist location(_, _::Event) = uniform_continuous(0, 1)
@dist time(_, _::Event) = uniform_continuous(0, 1)

@dist magnitude(_, _::Event) = 2 + exponential(log(10))

@gen (static) function arriving_log_amplitude(world, event::Event, station::Station)
    mag ~ lookup_or_generate(world[:magnitude][event])
    event_loc ~ lookup_or_generate(world[:location][event])
    stat_loc ~ lookup_or_generate(world[:station_location][station])
    return (mag - (α_0 * abs(event_loc - stat_loc)))
end

logistic(x, v, σ) = 1/(1 + exp(-(x-v)/σ))
@gen (static) function probability_event_detected_at_station(world, event::Event, station::Station)
    logamp ~ arriving_log_amplitude(world, event, station)
    noiselvl ~ lookup_or_generate(world[:background_noise_level][station])
    ν ~ lookup_or_generate(world[:ν][station])
    σ2 ~ lookup_or_generate(world[:σ2][station])
    return logistic(logamp - noiselvl, ν, sqrt(σ2)) # QUESTION: do we want to sqrt?
end

###################
# Detection model #
###################

@type Detection

@gen (static) function num_event_detections(world, tup::Tuple{Event, Station})
    (e, s) = tup
    prob ~ probability_event_detected_at_station(world, e, s)
    is_detected ~ bernoulli(prob)
    return is_detected
end

@gen (static) function num_noise_detections(world, s::Station)
    rate ~ lookup_or_generate(world[:false_alarm_rate][s])
    num ~ poisson(rate)
    return num
end

@gen (static) function measured_arrival_time(world, d::Detection)
    origin ~ lookup_or_generate(world[:origin][d])
    (event, station) = origin
    event_time ~ lookup_or_generate(world[:time][event])
    eventpos ~ lookup_or_generate(world[:location][event])
    statpos ~ lookup_or_generate(world[:station_location][station])
    travtime ~ travel_time(world, eventpos, statpos)
    (μ, σ2) = {:params} ~ lookup_or_generate(world[:arrival_time_measurement_error_params][()])
    obs ~ normal(event_time + travtime + μ, σ2)
    return obs
end

@gen (static) function measured_log_amplitude_real(world, d::Detection)
    origin ~ lookup_or_generate(world[:origin][d])
    (event, station) = origin
    arrivinglogamp ~ arriving_log_amplitude(world, event, station)
    (μ, σ2) = {:params} ~ lookup_or_generate(world[:amplitude_measurement_error_params][()])
    obs ~ normal(arrivinglogamp + μ, σ2)
    return obs
end
@gen (static) function real_detection_sign(world, d::Detection)
    origin ~ lookup_or_generate(world[:origin][d])
    (event, station) = origin
    eventpos ~ lookup_or_generate(world[:location][event])
    statpos ~ lookup_or_generate(world[:station_location][station])
    return sign(stationpos - eventpos)
end

@dist measured_false_detection_arrival_time(_, _::Detection) = uniform_continuous(0, 1)
@gen (static) function measured_log_amplitude_noise(world, d::Detection)
    origin ~ lookup_or_generate(world[:origin][d])
    (_, station) = origin
    params ~ lookup_or_generate(world[:noise_detection_amplitude_params][station])
    (μ, σ2) = params
    logamp ~ normal(μ, σ2)
    return logamp
end
@dist noise_sign(_, _::Detection) = 2*uniform_discrete(1, 2) - 3

struct Observation
    station_idx::Int # in 1...5
    log_amplitude::Float64
    time::Float64
    sign::Int # either 1 or -1
end

@gen function get_observation(world, detection)
    origin ~ lookup_or_generate(world[:origin][detection])
    is_noise = length(origin) == 1
    station = is_noise ? origin[1] : origin[2]
    stat_idx ~ lookup_or_generate(world[:index][station])

    if is_noise
        noise_sgn ~ lookup_or_generate(world[:noise_sign][detection])
        noise_time ~ lookup_or_generate(world[:measured_false_detection_arrival_time][detection])
        noise_amp ~ lookup_or_generate(world[:measured_log_amplitude_real][detection])
        return Observation(stat_idx, log_amp, noise_time, noise_sgn)
    else
        obs_sgn ~ lookup_or_generate(world[:real_detection_sign][detection])
        obs_time ~ lookup_or_generate(world[:measured_arrival_time][detection])
        obs_amp ~ lookup_or_generate(world[:measured_log_amplitude_noise][detection])
        return Observation(stat_idx, obs_amp, obs_time, obs_sgn)
    end
end

get_detections = GetOriginIteratingObjectSet(:Detection, (:num_noise_detections, :num_event_detections))

@gen (static) function _generate_observations(world)
    num_events ~ lookup_or_generate(world[:num_events][()])
    events ~ GetSingleOriginObjectSet(:Event)(world, (), num_events)
    stations ~ GetSingleOriginObjectSet(:Station)(world,(), NUM_STATIONS)
    detections ~ get_detections(world, (events,), (events, stations))
    observations ~ SetMap(lookup_or_generate)(mgfcall_setmap(world[:get_observation], detections))
    return observations
end

generate_observations = UsingWorld(
    _generate_observations,
    :event_occurance_rate => event_occurance_rate,
    :background_noise_level => background_noise_level,
    :false_alarm_rate => false_alarm_rate,
    :signal_velocity => signal_velocity,
    :absorpivity_per_unit_distance => absorpivity_per_unit_distance,
    :ν => ν,
    :σ2 => σ2,
    :arrival_time_measurement_error_params => arrival_time_measurement_error_params,
    :amplitude_measurement_error_params => amplitude_measurement_error_params,
    :noise_detection_amplitude_params => noise_detection_amplitude_params,
    :station_location => station_location,
    :travel_time => travel_time,
    :num_events => num_events,
    :location => location,
    :time => time,
    :num_event_detections => num_event_detections,
    :num_noise_detections => num_noise_detections,
    :measured_arrival_time => measured_arrival_time,
    :measured_log_amplitude_real => measured_log_amplitude_real,
    :real_detection_sign => real_detection_sign,
    :measured_false_detection_arrival_time => measured_false_detection_arrival_time,
    :measured_log_amplitude_noise => measured_log_amplitude_noise,
    :noise_sign => noise_sign,
    :get_observation => get_observation
)

@load_generated_functions()