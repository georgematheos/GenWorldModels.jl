@oupm generate_detections(num_stations) begin
    @type Event, Station, Detection

    @property magnitude(::Event) ~ exponential(1.0)
    is_false_positive(d::Detection) = length(@origin(d)) == 1
    @property function reading(d::Detection)
        if is_false_positive(d)
            reading ~ normal(0, 1)
        else
            (station, event) = @origin d
            reading ~ normal(@get magnitude[event], 0.4)
        end
        return reading
    end
    
    @number Station() = num_stations
    @number Event() ~ poisson(5)
    @number Detection(::Station) ~ poisson(4)
    @number Detection(::Station, ::Event) ~ int_bernoulli(0.8)

    @observation_model function detections()
        detections = @objects Detection
        return @map_get reading[detections]
    end
end

# would it be possible for something like defining
# is_false_positive to compile two methods, one for when
# `d` is abstract and one for when it is concrete?