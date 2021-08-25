@type Event
@type Station
@type Detection

expr = :(@oupm generate_detections(num_stations) begin
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
end)
# println(macroexpand(@__MODULE__, expr))
@eval $expr

@load_generated_functions()

# @testset "OUPM modeling DSL integration - seismic example" begin
#     num_constraints = choicemap(
#         (:world => Symbol("#Event()") => () => :num, 2),
#         (:world => Symbol("#Detection(Station)") => (Station(1),) => :num, 1),
#         (:world => Symbol("#Detection(Station)") => (Station(2),) => :num, 3),
#         (:world => Symbol("#Detection(Station, Event)") => (Station(1), Event(1)) => :num, 1),
#         (:world => Symbol("#Detection(Station, Event)") => (Station(2), Event(1)) => :num, 1),
#         (:world => Symbol("#Detection(Station, Event)") => (Station(1), Event(2)) => :num, 1),
#         (:world => Symbol("#Detection(Station, Event)") => (Station(2), Event(2)) => :num, 0),
#     )

#     @testset "object set construction" begin
#         tr, weight = generate(generate_detections, (2, :detections), num_constraints)
#         @test length(get_retval(tr)) == 7

#         tr, weight = generate(generate_detections, (2, :false_positives), num_constraints)
#         @test length(get_retval(tr)) == 4

#         # tr, weight = generate(generate_detections, (2, 1), num_constraints)
#         # @test length(get_retval(tr)) == 2

#         # tr, weight = generate(generate_detections, (2, 2), num_constraints)
#         # display(get_choices(tr))
#         # @test length(get_retval(tr)) == 1
#     end

#     @testset "simple value updating" begin
#         constraints = merge(choicemap(
#             (:world => :magnitude => (Event(1),), 1.0),
#             (:world => :magnitude => (Event(2),), 2.0),
#             (:world => :reading => (Detection((Station(1), Event(1)), 1),) => :reading, 1.5),
#             (:world => :reading => (Detection((Station(2), Event(1)), 1),) => :reading, 1.4),
#             (:world => :reading => (Detection((Station(1), Event(2)), 1),) => :reading, 2.3),
#         ), num_constraints)

#         tr, weight = generate(generate_detections, (2, :real_detections), constraints)
#         @test isapprox(get_score(tr), weight)

#         new_tr, weight, retdiff, discard = update(tr, (2, :real_detections), (NoChange(), NoChange()), choicemap(
#             (:world => :reading => (Detection((Station(1), Event(1)), 1),) => :reading, 1.0)
#         ))
#         @test 1.0 in get_retval(new_tr)
#         @test discard == choicemap((:world => :reading => (Detection((Station(1), Event(1)), 1),) => :reading, 1.5))
#         @test isapprox(weight, get_score(new_tr) - get_score(tr))
#         @test isapprox(weight, logpdf(normal, 1.0, 1.0, 0.4) - logpdf(normal, 1.5, 1.0, 0.4))

#         new_tr, weight, retdiff, discard = update(tr, (2, :real_detections), (NoChange(), NoChange()), choicemap(
#             (:world => :magnitude => (Event(1),), 2.0)
#         ))
#         @test discard == choicemap((:world => :magnitude => (Event(1),), 1.0))
#         @test isapprox(weight, logpdf(exponential, 2, 1) - logpdf(exponential, 1, 1) + logpdf(normal, 1.5, 2.0, 0.4) + logpdf(normal, 1.4, 2.0, 0.4) - (logpdf(normal, 1.5, 1.0, 0.4) + logpdf(normal, 1.4, 1.0, 0.4)))
#     end

#     @testset "updates changing looked up object set" begin
#         tr, weight = generate(generate_detections, (2, :real_detections), num_constraints)
        
#         new_tr, weight, retdiff, discard = update(tr, (2, :false_positives), (NoChange(), UnknownChange()), num_constraints)

#         @test length(get_retval(new_tr)) == 4

#         expected_discarded_detections = Set(
#             Detection(origin, 1) for origin in
#             ((Station(1), Event(1)), (Station(1), Event(2)), (Station(2), Event(1)))
#         )
#         @test all(
#             !isempty(get_subtree(discard, :world => :reading => (detection,)))
#             for detection in expected_discarded_detections
#         )
#     end
# end

# @oupm generate_values() begin
#     @number Station() ~ poisson(5)
#     @property val(::Station) ~ normal(0, 1)
#     @observation_model function get_vals()
#         stations = @objects Station
#         station1 ~ uniform_choice(stations)
#         station2 ~ uniform_choice(stations)

#         return [
#             (@map (@get val[stat] for stat in [station1, station2])),
#             (@map [@get val[stat] for (stat,) in [(station1,), (station2,)]]),
#             (@nocollision_setmap (@get val[stat] for stat in stations)),
#             (@nocollision_setmap [@get val[stat] for (stat,) in lazy_bijection_set_map(s -> (s,), (s,) -> s, stations)]),
#             (@dictmap (k => @get val[stat] for (k, stat) in lazy_set_to_dict_map(identity, stations))),
#             (@dictmap [(k => (@get val[stat])) for (k, stat) in lazy_set_to_dict_map(identity, stations)])
#         ]
#     end
# end
# @load_generated_functions()
# @testset "map syntaxes" begin
#     tr, _ = generate(generate_values, (), choicemap((:world => Symbol("#Station()") => () => :num, 5)))
#     new_tr, _, _, _ = update(tr, (), (), choicemap((:world => Symbol("#Station()") => () => :num, 6)))
    
#     for (r, size) in ((get_retval(tr), 5), (get_retval(new_tr), 6))
#         @test length(r) == 6
#         for v in (r[1], r[2])
#             @test v isa AbstractVector && all(e isa Real for e in v)
#             @test length(v) == 2
#         end
#         for s in (r[3], r[4])
#             @test s isa AbstractSet && all(e isa Real for e in s)
#             @test length(s) == size
#         end
#         for d in (r[5], r[6])
#             @test d isa AbstractDict && all(k isa Station && v isa Real for (k, v) in d)
#             @test length(d) == size
#         end

#     end
# end