@type Event
@type Station
@type Detection

expr = :(@oupm generate_detections(num_stations) begin
    @property magnitude(::Event) ~ exponential(1.0)
    @property is_false_positive(d::Detection) = length(@origin(d)) == 1
    @property function reading(d::Detection)
        if @get is_false_positive[d]
            reading ~ normal(0, 1)
        else
            (station, event) = @origin d
            reading ~ normal((@get magnitude[event]), 0.4)
        end
        return reading
    end
    
    @number Station() = @arg num_stations
    @number Event() ~ poisson(5)
    @number Detection(::Station) ~ poisson(4)
    @number Detection(::Station, ::Event) ~ bernoulli(0.8)

    @observation_model function detections()
        # detections = @objects Detection
        detections = [Detection((Station(1), ), 1)] # TODO: use @objects
        return @map [@get(reading[det]) for det in detections]
    end
end)

expr = macroexpand(@__MODULE__, expr, recursive=false)
# expr = macroexpand(@__MODULE__, expr, recursive=false)

expr = MacroTools.striplines(expr)
# Meta.show_sexpr(expr)
display(expr)
# println()
# println()
# for subexpr in expr.args
#     display(subexpr)
#     @eval $subexpr
#     display(macroexpand(@__MODULE__, subexpr, recursive=false))
#     println()
#     println()
# end
@eval $expr
println("done.")
@load_generated_functions()
# println(Event)
tr, weight = generate(generate_detections, (5,))
display(get_choices(tr))