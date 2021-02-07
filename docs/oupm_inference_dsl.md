Here's an example model:
```julia
@oupm generate_detections(num_stations) begin
    @number (static, diffs) Station() = (return @arg num_stations)
    @number Event() ~ poisson(5)
    @number Detection(::Station) ~ poisson(4)
    @number Detection(::Station, ::Event) ~ int_bernoulli(0.8)

    @property magnitude(::Event) ~ exponential(1.0)
    @property (static, diffs) is_false_positive(d::Detection) = (return length(@origin(d)) == 1)
    @property function reading(d::Detection)
        if @get is_false_positive[d]
            reading ~ normal(0, 1)
        else
            (station, event) = @origin d
            reading ~ normal((@get magnitude[event]), 0.4)
        end
        return reading
    end

    @observation_model function detections(set_to_lookup)
        if set_to_lookup === :detections
            detections = @objects Detection
        elseif set_to_lookup === :real_detections
            detections = @objects Detection(Station, Event)
        elseif set_to_lookup === :false_positives
            detections = @objects Detection(Station)
        # elseif set_to_lookup isa Int
        #     evt = Event(set_to_lookup)
        #     detections = @objects Detection(Station, evt)
        end

        return @nocollision_setmap (@get(reading[det]) for det in detections)
    end
end
```

Inference kernel:
```julia
is_detection_of(e, d) = !is_false_positive(d) && @origin(d)[2] == e
is_dubious(e, ds) = sum(is_detection_of(e, d) for d in ds) == 1
@gen function birth_death_proposal(prev_world)
dets, events, stations = [
    @objects(prev_world, T) for T in [Detection, Event, Station]
]

false_positives = filter(is_false_positive, dets)
dubious_events = filter(e -> is_dubious_event(e, dets), events)
create_prob = isempty(false_positives) ? 0.0 : isempty(dubious_events) ? 1.0 : 0.5
create_event ~ bernoulli(create_prob)


end
```