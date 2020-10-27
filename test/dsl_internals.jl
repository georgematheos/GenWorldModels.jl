using MacroTools
using GenWorldModels: OUPMDSLMetaData, parse_type_line!, parse_world_into_and_trace_commands
using GenWorldModels: parse_property_line!, parse_number_line!, OriginSignature

@testset "dsl parsing" begin
    @testset "parse type line" begin
        stmts = []
        meta = OUPMDSLMetaData(:name, ())
        parse_type_line!(stmts, meta, :(@type Event, Station, Detection))
        expected_names = Set((:Event, :Station, :Detection))

        names = []
        for stmt in stmts
            is_valid_type_call = MacroTools.@capture(stmt, @type Name_)
            push!(names, Name)
            @test is_valid_type_call
        end
        @test Set(names) == expected_names
        @test Set(meta.type_names) == expected_names
    end

    @testset "parse world into commands" begin
        body = quote
            (parent,) = @origin(object)
            val ~ normal(@get(value[parent]), 1)
            t = @time val + 5 # should not get transformed!
            (grandmother, grandfather) = @origin(@origin(object)[1])
        end

        expected = quote
            name1_ ~ @origin(world, object)
            (parent,) = name1_
            name2_ ~ @get(world, value[parent])
            val ~ normal(name2_, 1)
            t = @time val + 5
            name3_ ~ @origin(world, object)
            name4_ ~ @origin(world, name3_[1])
            (grandmother, grandfather) = name4_
        end

        transformed = parse_world_into_and_trace_commands(body, :world)

        @test MacroTools.@capture(transformed, $expected)
    end

    @testset "parse property line" begin
        @testset "dist property" begin
            stmts = []
            meta = OUPMDSLMetaData(:name, ())
            line = :(@property global_property() ~ normal(10, 1))
            expected = :(@dist global_property(::World) = normal(10, 1))
            parse_property_line!(stmts, meta, line)
            @test meta.property_names == Set([:global_property])
            @test length(stmts) == 1
            @test MacroTools.@capture(first(stmts), $expected)

            stmts = []
            meta = OUPMDSLMetaData(:name, ())
            line = :(@property magnitude(::Event) ~ normal(10, 1))
            expected = :(@dist magnitude(::Event, ::World) = normal(10, 1))
            parse_property_line!(stmts, meta, line)
            @test meta.property_names == Set([:magnitude])
            @test length(stmts) == 1
            @test MacroTools.@capture(first(stmts), $expected)

            stmts = []
            meta = OUPMDSLMetaData(:name, ())
            line = :(@property is_detected(::Station, evt::Event) ~ normal(10, 1))
            expected = :(@dist is_detected(::Station, evt::Event, ::World) = normal(10, 1))
            parse_property_line!(stmts, meta, line)
            @test meta.property_names == Set([:is_detected])
            @test length(stmts) == 1
            @test MacroTools.@capture(first(stmts), $expected)
        end

        @testset "full function declaration" begin
            ### inline def ###
            stmts = []
            meta = OUPMDSLMetaData(:name, ())
            line = :(@property (static) magnitude(evt::Event) = begin; mag ~ normal(0, 1); return mag; end;).args[1]

            expected = :(
                @gen (static) function magnitude(evt::Event, world_::World)
                    mag ~ normal(0, 1)
                    return mag
                end
            )

            parse_property_line!(stmts, meta, line)
            @test length(stmts) === 1
            @test MacroTools.@capture(first(stmts), $expected)
            @test meta.property_names == Set([:magnitude])

            stmts = []
            meta = OUPMDSLMetaData(:name, ())
            line = :(@property magnitude(evt::Event) = mag ~ normal(0, 1))

            expected = :(
                @gen function magnitude(evt::Event, world_::World)
                    mag ~ normal(0, 1)
                end
            )

            parse_property_line!(stmts, meta, line)
            @test length(stmts) === 1
            @test MacroTools.@capture(first(stmts), $expected)
            @test meta.property_names == Set([:magnitude])

            ### full def ###
            stmts = []
            meta = OUPMDSLMetaData(:name, ())
            line = :(
                @property (static, diffs) function reading(det::Detection)
                    reading ~ normal((@get magnitude[@origin(det)[2]]), 1)
                    return reading
                end
            )
            expected = :(
                @gen (static, diffs) function reading(det::Detection, world_::World)
                    og_ ~ @origin(world_, det)
                    mag_ ~ @get(world_, magnitude[og_[2]])
                    reading ~ normal(mag_, 1)
                    return reading
                end
            )
            parse_property_line!(stmts, meta, line)
            @test length(stmts) === 1
            @test MacroTools.@capture(first(stmts), $expected)
            @test meta.property_names == Set([:reading])

            stmts = []
            meta = OUPMDSLMetaData(:name, ())
            line = :(
                @property function reading(det::Detection)
                    reading ~ normal((@get magnitude[@origin(det)[2]]), 1)
                    return reading
                end
            )
            expected = :(
                @gen function reading(det::Detection, world_::World)
                    og_ ~ @origin(world_, det)
                    mag_ ~ @get(world_, magnitude[og_[2]])
                    reading ~ normal(mag_, 1)
                    return reading
                end
            )
            parse_property_line!(stmts, meta, line)
            @test length(stmts) === 1
            @test MacroTools.@capture(first(stmts), $expected)
            @test meta.property_names == Set([:reading])

        end
    end

    @testset "parse number statement" begin
        ### distribution number statement ###
        stmts = []
        meta = OUPMDSLMetaData(:name, ())
        line = :(@number Event() ~ poisson(5))
        expected = :(
            @gen (static, diffs) function numevt_(()::Tuple{}, ::World)
                num ~ poisson(5)
                return num
            end
        )
        parse_number_line!(stmts, meta, line)
        @test length(stmts) === 1

        @test MacroTools.@capture(first(stmts), $expected)
        @test haskey(meta.number_stmts, OriginSignature(:Event, ())) && length(meta.number_stmts) == 1

        stmts = []
        meta = OUPMDSLMetaData(:name, ())
        line = :(@number Detection(::Station) ~ poisson(5))
        expected = :(
            @gen (static, diffs) function numdet_((_,)::Tuple{Station}, ::World)
                num ~ poisson(5)
                return num
            end
        )
        parse_number_line!(stmts, meta, line)
        @test length(stmts) === 1
        @test MacroTools.@capture(first(stmts), $expected)
        @test haskey(meta.number_stmts, OriginSignature(:Detection, (:Station,))) && length(meta.number_stmts) == 1

        ### gen function num statement ###
        stmts = []
        meta = OUPMDSLMetaData(:name, ())
        line = :(@number Event() = num ~ poisson(5))

        expected = :(
            @gen function numevt_(()::Tuple{}, w_::World)
                num ~ poisson(5)
            end
        )
        parse_number_line!(stmts, meta, line)

        @test length(stmts) === 1
        @test MacroTools.@capture(first(stmts), $expected)
        @test haskey(meta.number_stmts, OriginSignature(:Event, ())) && length(meta.number_stmts) == 1

        stmts = []
        meta = OUPMDSLMetaData(:name, ())
        line = :(@number (static, diffs) Event() = begin; num ~ poisson(5); return num; end;)
        expected = :(
            @gen (static, diffs) function numevt_(()::Tuple{}, w_::World)
                num ~ poisson(5)
                return num
            end
        )
        parse_number_line!(stmts, meta, line)
        @test length(stmts) === 1

        @test MacroTools.@capture(first(stmts), $expected)
        @test haskey(meta.number_stmts, OriginSignature(:Event, ())) && length(meta.number_stmts) == 1

        stmts = []
        meta = OUPMDSLMetaData(:name, ())
        line = :(
            @number (static) function Detection(e::Event, s::Station)
                num ~ bernoulli(@get detection_prob[e])
                return num
            end
        )
        expected = :(
            @gen (static) function numdet_((e, s)::Tuple{Event, Station}, world_::World)
                prob_ ~ @get(world_, detection_prob[e])
                num ~ bernoulli(prob_)
                return num
            end
        )
        parse_number_line!(stmts, meta, line)
        @test length(stmts) === 1
        @test MacroTools.@capture(first(stmts), $expected)
        @test haskey(meta.number_stmts, OriginSignature(:Detection, (:Event, :Station))) && length(meta.number_stmts) == 1
        
        stmts = []
        meta = OUPMDSLMetaData(:name, ())
        line = :(
            @number function Detection(e::Event, s::Station)
                num ~ bernoulli(@get detection_prob[e])
                return num
            end
        )
        expected = :(
            @gen function numdet_((e, s)::Tuple{Event, Station}, world_::World)
                prob_ ~ @get(world_, detection_prob[e])
                num ~ bernoulli(prob_)
                return num
            end
        )
        parse_number_line!(stmts, meta, line)
        @test length(stmts) === 1
        @test MacroTools.@capture(first(stmts), $expected)
        @test haskey(meta.number_stmts, OriginSignature(:Detection, (:Event, :Station))) && length(meta.number_stmts) == 1
    end
end