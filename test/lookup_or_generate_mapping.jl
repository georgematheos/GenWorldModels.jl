@dist size(::World, ::Aircraft) = exponential(100)
@dist num_aircrafts_small(::World, ::Tuple{}) = poisson(5)
@gen (static, diffs) function _map_get_sizes(world)
    n_a ~ lookup_or_generate(world[:num_aircrafts][()])
    aircrafts = [Aircraft(i) for i=1:n_a]
    sizes ~ map_lookup_or_generate(world[:size], aircrafts)
    return sizes
end
@gen (static, diffs) function _nocollision_setmap_get_sizes(world)
    aircrafts ~ get_sibling_set(:Aircraft, :num_aircrafts, world, ())
    sizes ~ nocollision_setmap_lookup_or_generate(world[:size], aircrafts)
    return sizes
end
@gen (static, diffs) function _setmap_get_sizes(world)
    aircrafts ~ get_sibling_set(:Aircraft, :num_aircrafts, world, ())
    sizes ~ nocollision_setmap_lookup_or_generate(world[:size], aircrafts)
    return sizes
end
@gen (static, diffs) function _dictmap_get_sizes(world)
    num_aircrafts ~ lookup_or_generate(world[:num_aircrafts][()])
    idx_to_aircraft = Dict(i => Aircraft(i) for i=1:num_aircrafts)
    idx_to_size ~ dictmap_lookup_or_generate(world[:size], idx_to_aircraft)
    return idx_to_size
end
@load_generated_functions()

@testset "mapped lookup_or_generate" begin
    @testset "map_lookup_or_generate" begin
        get_sizes = UsingWorld(_map_get_sizes, :size => size, :num_aircrafts => num_aircrafts_small)

        tr, weight = generate(get_sizes, (), choicemap((:world => :num_aircrafts => (), 5)))
        @test length(get_retval(tr)) == 5
        @test length(unique(get_retval(tr))) == 5

        firstval = get_retval(tr)[1]
        constraint_small = choicemap((:world => :size => Aircraft(1), firstval + 1))
        new_tr, weight, retdiff, discard = update(tr, (), (), constraint_small)
        @test new_tr[:world => :size => Aircraft(1)] == firstval + 1
        @test isapprox(weight, logpdf(exponential, firstval + 1, 100) - logpdf(exponential, firstval, 100))
        @test retdiff isa VectorDiff
        @test retdiff.prev_length == 5 && retdiff.new_length == 5
        @test retdiff.updated == Dict(1 => UnknownChange())
        @test discard == choicemap((:world => :size => Aircraft(1), firstval))

        # test that the time for updating scales approximately constantly in the size of the world
        bigtr, _ = generate(get_sizes, (), choicemap((:world => :num_aircrafts => (), 1000)))

        smalltime = 0
        bigtime = 0
        for _=1:4
            smalltime += @elapsed update(tr, (), (), constraint_small)
            bigtime += @elapsed update(tr, (), (), constraint_small)
        end
        @test bigtime < 3*smalltime
    end

    @testset "nocollision_setmap_lookup_or_generate" begin
        get_sizes = UsingWorld(_nocollision_setmap_get_sizes, :size => size, :num_aircrafts => num_aircrafts_small)

        tr, weight = generate(get_sizes, (), choicemap((:world => :num_aircrafts => (), 5)))
        @test length(get_retval(tr)) == 5
        @test length(unique(get_retval(tr))) == 5

        firstval = tr[:world => :size => Aircraft(1)]
        constraint_small = choicemap((:world => :size => Aircraft(1), firstval + 1))
        new_tr, weight, retdiff, discard = update(tr, (), (), constraint_small)
        @test new_tr[:world => :size => Aircraft(1)] == firstval + 1
        @test isapprox(weight, logpdf(exponential, firstval + 1, 100) - logpdf(exponential, firstval, 100))
        @test retdiff isa SetDiff
        @test retdiff.added == Set([firstval + 1])
        @test retdiff.deleted == Set([firstval])
        @test discard == choicemap((:world => :size => Aircraft(1), firstval))

        # test that the time for updating scales approximately constantly in the size of the world
        bigtr, _ = generate(get_sizes, (), choicemap((:world => :num_aircrafts => (), 1000)))

        smalltime = 0
        bigtime = 0
        for _=1:4
            smalltime += @elapsed update(tr, (), (), constraint_small)
            bigtime += @elapsed update(tr, (), (), constraint_small)
        end
        @test bigtime < 3*smalltime
    end
   
    @testset "setmap_lookup_or_generate" begin
        get_sizes = UsingWorld(_setmap_get_sizes, :size => size, :num_aircrafts => num_aircrafts_small)

        tr, weight = generate(get_sizes, (), choicemap((:world => :num_aircrafts => (), 5)))
        @test length(get_retval(tr)) == 5
        @test length(unique(get_retval(tr))) == 5

        firstval = tr[:world => :size => Aircraft(1)]
        constraint_small = choicemap((:world => :size => Aircraft(1), firstval + 1))
        new_tr, weight, retdiff, discard = update(tr, (), (), constraint_small)
        @test new_tr[:world => :size => Aircraft(1)] == firstval + 1
        @test isapprox(weight, logpdf(exponential, firstval + 1, 100) - logpdf(exponential, firstval, 100))
        @test retdiff isa SetDiff
        @test retdiff.added == Set([firstval + 1])
        @test retdiff.deleted == Set([firstval])
        @test discard == choicemap((:world => :size => Aircraft(1), firstval))

        # test that the time for updating scales approximately constantly in the size of the world
        bigtr, _ = generate(get_sizes, (), choicemap((:world => :num_aircrafts => (), 1000)))

        smalltime = 0
        bigtime = 0
        for _=1:4
            smalltime += @elapsed update(tr, (), (), constraint_small)
            bigtime += @elapsed update(tr, (), (), constraint_small)
        end
        @test bigtime < 3*smalltime
    end

    @testset "dictmap_lookup_or_generate" begin
        get_sizes = UsingWorld(_dictmap_get_sizes, :size => size, :num_aircrafts => num_aircrafts_small)

        tr, weight = generate(get_sizes, (), choicemap((:world => :num_aircrafts => (), 5)))
        @test length(get_retval(tr)) == 5
        @test length(unique(get_retval(tr))) == 5

        firstval = tr[:world => :size => Aircraft(1)]
        constraint_small = choicemap((:world => :size => Aircraft(1), firstval + 1))
        new_tr, weight, retdiff, discard = update(tr, (), (), constraint_small)
        @test new_tr[:world => :size => Aircraft(1)] == firstval + 1
        @test isapprox(weight, logpdf(exponential, firstval + 1, 100) - logpdf(exponential, firstval, 100))
        @test retdiff isa DictDiff
        @test isempty(retdiff.added)
        @test isempty(retdiff.deleted)
        @test retdiff.updated == Dict(1 => UnknownChange())
        @test discard == choicemap((:world => :size => Aircraft(1), firstval))

        # test that the time for updating scales approximately constantly in the size of the world
        bigtr, _ = generate(get_sizes, (), choicemap((:world => :num_aircrafts => (), 1000)))

        smalltime = 0
        bigtime = 0
        for _=1:4
            smalltime += @elapsed update(tr, (), (), constraint_small)
            bigtime += @elapsed update(tr, (), (), constraint_small)
        end
        @test bigtime < 3*smalltime
    end
end