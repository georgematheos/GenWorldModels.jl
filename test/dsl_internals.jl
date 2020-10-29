using MacroTools
using GenWorldModels: OUPMDSLMetaData #, parse_world_into_and_trace_commands
using GenWorldModels: parse_property_line!, parse_number_line!, OriginSignature, parse_observation_model!

# TODO: update these tests once the internal scheme for macroexpansion has stabilized a bit

# @testset "dsl parsing" begin
#     @testset "parse world into commands" begin
#         body = quote
#             (parent,) = @origin(object)
#             val ~ normal(@get(value[parent]), 1)
#             t = @time val + 5 # should not get transformed!
#             (grandmother, grandfather) = @origin(@origin(object)[1])
#         end

#         expected = quote
#             (parent,) = ({name1_} ~ @origin(world, object))
#             val ~ normal({name2_} ~ @get(world, value[parent]), 1)
#             t = @time val + 5
#             (grandmother, grandfather) = (
#                 {name4_} ~ @origin(world,
#                     ({name3_} ~ @origin(world, object))[1]
#                 )
#             )
#         end

#         transformed = parse_world_into_and_trace_commands(body, :world)

#         @test MacroTools.@capture(transformed, $expected)
#     end

#     @testset "parse property line" begin
#         @testset "dist property" begin
#             stmts = []
#             meta = OUPMDSLMetaData(:name, ())
#             line = :(@property global_property() ~ normal(10, 1))
#             expected = :(@dist global_property(::World, ()::Tuple{}) = normal(10, 1))
#             parse_property_line!(stmts, meta, line)

#             @test meta.property_names == Set([:global_property])
#             @test length(stmts) == 1
#             @test MacroTools.@capture(first(stmts), $expected)

#             stmts = []
#             meta = OUPMDSLMetaData(:name, ())
#             line = :(@property magnitude(::Event) ~ normal(10, 1))
#             expected = :(@dist magnitude(::World, (_,)::Tuple{Event}) = normal(10, 1))
#             parse_property_line!(stmts, meta, line)
#             @test meta.property_names == Set([:magnitude])
#             @test length(stmts) == 1
#             @test MacroTools.@capture(first(stmts), $expected)

#             stmts = []
#             meta = OUPMDSLMetaData(:name, ())
#             line = :(@property is_detected(::Station, evt::Event) ~ normal(10, 1))
#             expected = :(@dist is_detected(::World, (_, evt)::Tuple{Station, Event}) = normal(10, 1))
#             parse_property_line!(stmts, meta, line)
#             @test meta.property_names == Set([:is_detected])
#             @test length(stmts) == 1
#             @test MacroTools.@capture(first(stmts), $expected)
#         end

#         @testset "full function declaration" begin
#             ### inline def ###
#             stmts = []
#             meta = OUPMDSLMetaData(:name, ())
#             line = :(@property (static) magnitude(evt::Event) = begin; mag ~ normal(0, 1); return mag; end;).args[1]

#             expected = :(
#                 @gen (static) function magnitude(world_::World, (evt,)::Tuple{Event})
#                     mag ~ normal(0, 1)
#                     return mag
#                 end
#             )

#             parse_property_line!(stmts, meta, line)
#             @test length(stmts) === 1
#             @test MacroTools.@capture(first(stmts), $expected)
#             @test meta.property_names == Set([:magnitude])

#             stmts = []
#             meta = OUPMDSLMetaData(:name, ())
#             line = :(@property magnitude(evt::Event) = mag ~ normal(0, 1))

#             expected = :(
#                 @gen function magnitude(world_::World, (evt,)::Tuple{Event})
#                     mag ~ normal(0, 1)
#                 end
#             )

#             parse_property_line!(stmts, meta, line)
#             @test length(stmts) === 1
#             @test MacroTools.@capture(first(stmts), $expected)
#             @test meta.property_names == Set([:magnitude])

#             ### full def ###
#             stmts = []
#             meta = OUPMDSLMetaData(:name, ())
#             line = :(
#                 @property (static, diffs) function reading(det::Detection)
#                     reading ~ normal((@get magnitude[@origin(det)[2]]), 1)
#                     return reading
#                 end
#             )
#             expected = :(
#                 @gen (static, diffs) function reading(world_::World, (det,)::Tuple{Detection})
#                     reading ~ normal(
#                         ({name1_} ~ @get(world_, magnitude[
#                             ({name2_} ~ @origin(world_, det))[2]
#                         ])),
#                         1
#                     )
#                     return reading
#                 end
#             )
#             parse_property_line!(stmts, meta, line)
#             @test length(stmts) === 1
#             @test MacroTools.@capture(first(stmts), $expected)
#             @test meta.property_names == Set([:reading])

#             stmts = []
#             meta = OUPMDSLMetaData(:name, ())
#             line = :(
#                 @property function reading(det::Detection)
#                     reading ~ normal((@get magnitude[@origin(det)[2]]), 1)
#                     return reading
#                 end
#             )
#             expected = :(
#                 @gen function reading(world_::World, (det,)::Tuple{Detection})
#                     reading ~ normal(
#                         ({name1_} ~ @get(world_, magnitude[
#                             ({name2_} ~ @origin(world_, det))[2]
#                         ])),
#                         1
#                     )
#                     return reading
#                 end
#             )
#             parse_property_line!(stmts, meta, line)
#             @test length(stmts) === 1
#             @test MacroTools.@capture(first(stmts), $expected)
#             @test meta.property_names == Set([:reading])

#         end
#     end

#     @testset "parse number statement" begin
#         ### distribution number statement ###
#         stmts = []
#         meta = OUPMDSLMetaData(:name, ())
#         line = :(@number Event() ~ poisson(5))
#         expected = :(
#             @gen (static, diffs) function numevt_(::World, ()::Tuple{})
#                 num ~ poisson(5)
#                 return num
#             end
#         )
#         parse_number_line!(stmts, meta, line)
#         @test length(stmts) === 1

#         @test MacroTools.@capture(first(stmts), $expected)
#         @test haskey(meta.number_stmts, OriginSignature(:Event, ())) && length(meta.number_stmts) == 1

#         stmts = []
#         meta = OUPMDSLMetaData(:name, ())
#         line = :(@number Detection(::Station) ~ poisson(5))
#         expected = :(
#             @gen (static, diffs) function numdet_(::World, (_,)::Tuple{Station})
#                 num ~ poisson(5)
#                 return num
#             end
#         )
#         parse_number_line!(stmts, meta, line)
#         @test length(stmts) === 1
#         @test MacroTools.@capture(first(stmts), $expected)
#         @test haskey(meta.number_stmts, OriginSignature(:Detection, (:Station,))) && length(meta.number_stmts) == 1

#         ### gen function num statement ###
#         stmts = []
#         meta = OUPMDSLMetaData(:name, ())
#         line = :(@number Event() = num ~ poisson(5))

#         expected = :(
#             @gen function numevt_(w_::World, ()::Tuple{})
#                 num ~ poisson(5)
#             end
#         )
#         parse_number_line!(stmts, meta, line)

#         @test length(stmts) === 1
#         @test MacroTools.@capture(first(stmts), $expected)
#         @test haskey(meta.number_stmts, OriginSignature(:Event, ())) && length(meta.number_stmts) == 1

#         stmts = []
#         meta = OUPMDSLMetaData(:name, ())
#         line = :(@number (static, diffs) Event() = begin; num ~ poisson(5); return num; end;)
#         expected = :(
#             @gen (static, diffs) function numevt_(w_::World, ()::Tuple{})
#                 num ~ poisson(5)
#                 return num
#             end
#         )
#         parse_number_line!(stmts, meta, line)
#         @test length(stmts) === 1

#         @test MacroTools.@capture(first(stmts), $expected)
#         @test haskey(meta.number_stmts, OriginSignature(:Event, ())) && length(meta.number_stmts) == 1

#         stmts = []
#         meta = OUPMDSLMetaData(:name, ())
#         line = :(
#             @number (static) function Detection(e::Event, s::Station)
#                 num ~ bernoulli(@get detection_prob[e])
#                 return num
#             end
#         )
#         expected = :(
#             @gen (static) function numdet_(world_::World, (e, s)::Tuple{Event, Station})
#                 num ~ bernoulli({prob_} ~ @get(world_, detection_prob[e]))
#                 return num
#             end
#         )
#         parse_number_line!(stmts, meta, line)
#         @test length(stmts) === 1
#         @test MacroTools.@capture(first(stmts), $expected)
#         @test haskey(meta.number_stmts, OriginSignature(:Detection, (:Event, :Station))) && length(meta.number_stmts) == 1
        
#         stmts = []
#         meta = OUPMDSLMetaData(:name, ())
#         line = :(
#             @number function Detection(e::Event, s::Station)
#                 num ~ bernoulli(@get detection_prob[e])
#                 return num
#             end
#         )
#         expected = :(
#             @gen function numdet_(world_::World, (e, s)::Tuple{Event, Station})
#                 num ~ bernoulli({prob_} ~ @get(world_, detection_prob[e]))
#                 return num
#             end
#         )
#         parse_number_line!(stmts, meta, line)
#         @test length(stmts) === 1
#         @test MacroTools.@capture(first(stmts), $expected)
#         @test haskey(meta.number_stmts, OriginSignature(:Detection, (:Event, :Station))) && length(meta.number_stmts) == 1
#     end

#     @testset "parse observation model" begin
#         stmts = []
#         meta = OUPMDSLMetaData(:name, ())
#         line = :(@observation_model function foo(x::Int, y::Float)
#                     z ~ normal(x, y)
#                     return z
#                 end
#         )
#         expected = :(
#             @gen function foo(world_::World, x::Int, y::Float)
#                 z ~ normal(x, y)
#                 return z
#             end
#         )

#         parse_observation_model!(stmts, meta, line)

#         @test length(stmts) === 1
#         @test MacroTools.@capture(first(stmts), $expected)
#         @test meta.observation_model_name == :foo
#         @test_throws Exception parse_observation_model!(stmts, meta, line)

#         stmts = []
#         meta = OUPMDSLMetaData(:name, ())
#         line = :(@observation_model (static) function foo(x::Int, y::Float)
#                     z ~ normal(x, y)
#                     return z
#                 end
#         )
#         expected = :(
#             @gen (static) function foo(world_::World, x::Int, y::Float)
#                 z ~ normal(x, y)
#                 return z
#             end
#         )

#         parse_observation_model!(stmts, meta, line)
#         @test length(stmts) === 1
#         @test MacroTools.@capture(first(stmts), $expected)
#         @test meta.observation_model_name == :foo
#     end
# end