using MacroTools
using GenWorldModels: OUPMDSLMetaData, parse_type_line!, parse_world_into_and_trace_commands, parse_property_line!

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
            stmts = []
            meta = OUPMDSLMetaData(:name, ())
            

        end
    end
end