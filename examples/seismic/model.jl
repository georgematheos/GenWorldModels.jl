#=
TODOs:
- Get Dist dsl working
- Inverse normal gamma distribution
=#

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
@dist background_noise_level(_, ::Station) = inv_gamma(α_N, β_N)
@dist false_alarm_rate(_, ::Station) = gamma(α_F, β_F)
@dist function signal_velocity(_, _)
    W_0 = normal(μ_V, σ2_V)
    return W_0^2
end
@gen (static) function absorpivity_per_unit_distance(_, _)
    β_0 ~ normal(μ_B, σ2_B)
    return β_0^2
end
@dist ν(_, ::Station) = normal(μ_v, σ2_v)
@dist σ2(_, ::Station) = inv_gamma(α_s, β_s)
@dist arrival_time_measurement_error_params(_, ::Station) = normal_inv_gamma(μ_t, λ_t, α_t, β_t)
@dist amplitude_measurement_error_params(_, ::Station) = normal_inv_gamma(μ_a, λ_a, α_a, β_a)
@dist noise_detection_amplitude_params(_, ::Station) = norm_inv_gamma(μ_n, λ_n, α_n, β_n)

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

@gen (static) function num_events(world, _)
    rate ~ lookup_or_generate(world[:event_occurance_rate][()])
    num ~ poisson(rate)
    return num
end

@dist location(_, ::Event) = uniform_continuous(0, 1)
@dist time(_, ::Event) = uniform_continuous(0, 1)

# TODO: magnitude(_, ::Event)

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

@dist measured_false_detection_arrival_time(_, ::Detection) = uniform_continuous(0, 1)
@gen (static) function measured_log_amplitude_noise(world, d::Detection)
    origin ~ lookup_or_generate(world[:origin][d])
    (_, station) = origin
    params ~ lookup_or_generate(world[:noise_detection_amplitude_params][station])
    (μ, σ2) = params
    logamp ~ normal(μ, σ2)
    return logamp
end
@dist function noise_sign(_, ::Detection)
    x ~ uniform_discrete(1, 2)
    return (x*2) - 3
end

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

get_detections = OriginIteratingObjectSet(
    (event,) => :world => :num_noise_detections => (),
    (event, station) => :world => num_event_detections => (event, station)
)

@gen (static) function generate_observations(world)
    num_events ~ lookup_or_generate(world[:num_events][()])
    events ~ OriginlessObjectSet(Event)(num_events)
    stations ~ OriginlessObjectSet(Station)(NUM_STATIONS)
    detections ~ get_detections(world, (events,), (events, stations))
    observations ~ ObjectSetMap(:get_observation)(world, detections)
    return observations
end