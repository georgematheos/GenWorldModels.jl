@gen num_aircrafts(_::World, _::Tuple{}) = {:num} ~ poisson(5)
@gen num_blips(_::World, _::Tuple{<:Aircraft, <:Timestep}) = {:num} ~ poisson(1)
@gen plane_size(_::World, _::Aircraft) = {:size} ~ normal(600, 100)
@gen (static) function blip_size(world, b::Blip)
    (aircraft, timestep) = {:origin} ~ lookup_or_generate(world[:origin][b])
    plane_size ~ lookup_or_generate(world[:plane_size][aircraft])
    blip_size ~ normal(plane_size/600, 0.05)
    return blip_size
end

# returns a dict of (aircraft_idx, blip_idx_for_time_and_aircraft) => blip_size for every blip at this time
@gen (static) function blip_sizes_at_time(world, time)
    n_a ~ lookup_or_generate(world[:num_aircrafts][()])
    blips_per_aircraft ~ Map(lookup_or_generate)([world[:num_blips][(Aircraft(i), time)] for i=1:n_a])
    pairs = [(i, j) for i=1:n_a for j=1:blips_per_aircraft[i]]
    blips = [Blip((Aircraft(i), time), j) for (i, j) in pairs]
    sizes ~ Map(lookup_or_generate)([world[:blip_size][blip] for blip in blips])
    return Dict([pair => size for (pair, size) in zip(pairs, sizes)])
end

@gen (static) function _get_blip_sizes_at_times(world, timesteps)
    blips_sizes_per_time ~ Map(blip_sizes_at_time)(fill(world, length(timesteps)), [Timestep(i) for i in timesteps])
    return Dict([
        Blip((Aircraft(i), Timestep(timestep)), j) => size
        for (timestep, dict) in zip(timesteps, blips_sizes_per_time)
        for ((i, j), size) in dict
    ])
end
get_blip_sizes_at_times = UsingWorld(_get_blip_sizes_at_times,
    :num_aircrafts => num_aircrafts, :num_blips => num_blips, :plane_size => plane_size, :blip_size => blip_size #=;
    nums_statements=(
        NumStatement(Blip, (Aircraft, Timestep), :num_blips),
        NumStatement(Aircraft, (), :num_aircrafts)
    ) =#
)
@load_generated_functions()

@testset "OUPM moves for origin objects" begin
    constraints = choicemap(
        (:world => :num_aircrafts => () => :num, 2),
        (:world => :num_blips => (Aircraft(1), Timestep(1)) => :num, 1),
        (:world => :num_blips => (Aircraft(2), Timestep(1)) => :num, 1),
        (:world => :num_blips => (Aircraft(1), Timestep(2)) => :num, 1),
        (:world => :num_blips => (Aircraft(2), Timestep(2)) => :num, 1),
    )
    tr, _ = generate(get_blip_sizes_at_times, ([1, 2],), constraints)
    @test get_retval(tr) isa Dict{ConcreteIndexOUPMObject{:Blip, Tuple{ConcreteIndexOUPMObject{:Aircraft, Tuple{}}, ConcreteIndexOUPMObject{:Timestep, Tuple{}}}}, Float64}

    @testset "movemove" begin
        tr, _ = generate(get_blip_sizes_at_times, ([1, 2],), constraints)
        spec = UpdateWithOUPMMovesSpec(
            (
                MoveMove(
                    Blip((Aircraft(2), Timestep(1)), 1),
                    Blip((Aircraft(2), Timestep(2)), 1)
                ),
            ),
            choicemap(
                (:world => :num_blips => (Aircraft(2), Timestep(1)) => :num, 0),
                (:world => :num_blips => (Aircraft(2), Timestep(2)) => :num, 2)
            )
        )
        new_tr, weight, retdiff, rev = update(tr, ([1, 2],), (NoChange(),), spec, AllSelection())
        new_dict = get_retval(new_tr)
        old_dict = get_retval(tr)

        # println("old num blips:")
        # display(get_submap(get_choices(tr), :world => :num_blips))

        # println("new num blips:")
        # display(get_submap(get_choices(new_tr), :world => :num_blips))

        # display(old_dict)
        # display(tr.world.id_table)

        # println()
        # display(new_dict)
        # display(new_tr.world.id_table)

        @test old_dict[Blip((Aircraft(2), Timestep(1)), 1)] == new_dict[Blip((Aircraft(2), Timestep(2)), 1)]
        @test old_dict[Blip((Aircraft(2), Timestep(2)), 1)] == new_dict[Blip((Aircraft(2), Timestep(2)), 2)]
        @test !haskey(new_dict, Blip((Aircraft(2), Timestep(1)), 1))
        expected_score_diff = logpdf(poisson, 2, 1) + logpdf(poisson, 0, 1) - (2*logpdf(poisson, 1, 1))
        @test isapprox(weight, expected_score_diff)
        @test isapprox(get_score(new_tr) - get_score(tr), expected_score_diff)

        @test rev isa UpdateWithOUPMMovesSpec
        @test rev.moves == (MoveMove(Blip((Aircraft(2), Timestep(2)), 1), Blip((Aircraft(2), Timestep(1)), 1)),)
        @test rev.subspec == choicemap(
            (:world => :num_blips => (Aircraft(2), Timestep(1)) => :num, 1),
            (:world => :num_blips => (Aircraft(2), Timestep(2)) => :num, 1)
        )
    end

    @testset "birth moves" begin
        tr, _ = generate(get_blip_sizes_at_times, ([1, 2],), constraints)

        # birth of object with no origins
        spec = UpdateWithOUPMMovesSpec(
            (
                BirthMove(Blip((Aircraft(2), Timestep(1)), 1),),
            ),
            choicemap(
                (:world => :num_blips => (Aircraft(2), Timestep(1)) => :num, 2),
            )
        )
        new_tr, weight, retdiff, rev = update(tr, ([1, 2],), (NoChange(),), spec, AllSelection())
        new_dict = get_retval(new_tr)
        old_dict = get_retval(tr)

        @test old_dict[Blip((Aircraft(2), Timestep(1)), 1)] == new_dict[Blip((Aircraft(2), Timestep(1)), 2)]
        @test old_dict[Blip((Aircraft(2), Timestep(1)), 1)] != new_dict[Blip((Aircraft(2), Timestep(1)), 1)]
        newsize = new_dict[Blip((Aircraft(2), Timestep(1)), 1)]
        new_dict[Blip((Aircraft(2), Timestep(1)), 1)] = new_dict[Blip((Aircraft(2), Timestep(1)), 2)]
        delete!(new_dict, Blip((Aircraft(2), Timestep(1)), 2))
        @test old_dict == new_dict
        expected_weight = logpdf(poisson, 2, 1) - logpdf(poisson, 1, 1)
        expected_score_diff = expected_weight + logpdf(normal, newsize, new_tr[:world => :plane_size => Aircraft(2)]/600, 0.05)

        # display(get_choices(tr))
        # display(get_choices(new_tr))

        @test isapprox(weight, expected_weight)
        @test isapprox(get_score(new_tr) - get_score(tr), expected_score_diff)
        @test rev isa UpdateWithOUPMMovesSpec
        @test rev.moves == (DeathMove(Blip((Aircraft(2), Timestep(1)), 1)),)
        @test rev.subspec == choicemap((:world => :num_blips => (Aircraft(2), Timestep(1)) => :num, 1),)

        # birth of object with origins
        spec = UpdateWithOUPMMovesSpec(
            (
                BirthMove(Aircraft(1),),
            ),
            choicemap(
                (:world => :num_aircrafts => () => :num, 3),
            )
        )
        new_tr, weight, retdiff, rev = update(tr, ([1, 2],), (NoChange(),), spec, AllSelection())
        new_dict = get_retval(new_tr)
        old_dict = get_retval(tr)
        newdict2 = Dict()

        newvals = []
        for (blip, size) in new_dict
            aidx = blip.origin[1].idx
            if aidx > 1
                newdict2[Blip((Aircraft(aidx - 1), blip.origin[2]), blip.idx)] = size
            else
                push!(newvals, size)
            end
        end
        @test newdict2 == old_dict
        @test new_tr[:world => :plane_size => Aircraft(2)] == tr[:world => :plane_size => Aircraft(1)]
        @test new_tr[:world => :plane_size => Aircraft(3)] == tr[:world => :plane_size => Aircraft(2)]

        expected_weight = logpdf(poisson, 3, 5) - logpdf(poisson, 2, 5)
        expected_score_diff = expected_weight
        nb1 = new_tr[:world => :num_blips => (Aircraft(1), Timestep(1))]
        nb2 = new_tr[:world => :num_blips => (Aircraft(1), Timestep(2))]
        if nb1 + nb2 > 0 # if we have any blips for the new aircraft 1, we generated a plane size
            expected_score_diff += logpdf(normal, new_tr[:world => :plane_size => Aircraft(1)], 600, 100)
        end
        expected_score_diff += logpdf(poisson, nb1, 1)
        expected_score_diff += logpdf(poisson, nb2, 1)
        for val in newvals
            expected_score_diff += logpdf(normal, val, new_tr[:world => :plane_size => Aircraft(1)]/600, .05)
        end
        @test isapprox(get_score(new_tr) - get_score(tr), expected_score_diff)
        @test isapprox(weight, expected_weight)
        
        @test rev isa UpdateWithOUPMMovesSpec
        @test rev.moves == (DeathMove(Aircraft(1)),)
        @test rev.subspec == choicemap((:world => :num_aircrafts => () => :num, 2))
    end

    @testset "death moves" begin
        tr, _ = generate(get_blip_sizes_at_times, ([1, 2],), constraints)

        # death of object with no origins
        spec = UpdateWithOUPMMovesSpec(
            (
                DeathMove(Blip((Aircraft(2), Timestep(1)), 1),),
            ),
            choicemap(
                (:world => :num_blips => (Aircraft(2), Timestep(1)) => :num, 0),
            )
        )
        new_tr, weight, retdiff, rev = update(tr, ([1, 2],), (NoChange(),), spec, AllSelection())
        new_dict = get_retval(new_tr)
        old_dict = deepcopy(get_retval(tr))
        size = pop!(old_dict, Blip((Aircraft(2), Timestep(1)), 1))
        @test old_dict == new_dict
        expected_weight = logpdf(poisson, 0, 1) - logpdf(poisson, 1, 1)
        expected_weight -= logpdf(normal, size, tr[:world => :plane_size => Aircraft(2)]/600, 0.05)

        @test isapprox(weight, expected_weight)
        @test isapprox(get_score(new_tr) - get_score(tr), expected_weight)
        @test rev isa UpdateWithOUPMMovesSpec
        @test rev.moves == (BirthMove(Blip((Aircraft(2), Timestep(1)), 1)),)
        @test rev.subspec == choicemap(
            (:world => :blip_size => Blip((Aircraft(2), Timestep(1)), 1) => :blip_size, size),
            (:world => :num_blips => (Aircraft(2), Timestep(1)) => :num, 1)
        )

        # this should delete Aircraft(1) and all blips it is an origin for; it should move down Aircraft(2) to Aircraft(1)
        # and have all the blips with Aircraft(1) as origin adjust down
        spec = UpdateWithOUPMMovesSpec(
            (
                DeathMove(Aircraft(1)),
            ),
            choicemap(
                (:world => :num_aircrafts => () => :num, 1),
            )
        )
        new_tr, weight, retdiff, rev = update(tr, ([1, 2],), (NoChange(),), spec, AllSelection())
        new_dict = get_retval(new_tr)
        old_dict = get_retval(tr)
        olddict_shifted = Dict()
        removed_vals = []
        discarded_vals = choicemap(
            (:world => :plane_size => Aircraft(1) => :size, tr[:world => :plane_size => Aircraft(1)]),
            (:world => :num_aircrafts => () => :num, 2),
            (:world => :num_blips => (Aircraft(1), Timestep(1)) => :num, 1),
            (:world => :num_blips => (Aircraft(1), Timestep(2)) => :num, 1)
        )
        for (blip, size) in old_dict
            aidx = blip.origin[1].idx
            if aidx > 1
                olddict_shifted[Blip((Aircraft(aidx - 1), blip.origin[2]), blip.idx)] = size
            else
                push!(removed_vals, size)
                discarded_vals[:world => :blip_size => blip => :blip_size] = size
            end
        end
        @test olddict_shifted == new_dict

        expected_score_diff = (
            logpdf(poisson, 1, 5)
            - logpdf(poisson, 2, 5)
            - logpdf(normal, tr[:world => :plane_size => Aircraft(1)], 600, 100)
            - 2*logpdf(poisson, 1, 1)
            - sum(
                logpdf(normal, val, tr[:world => :plane_size => Aircraft(1)]/600, 0.05)
                for val in removed_vals
            )
        )
        @test isapprox(get_score(new_tr) - get_score(tr), expected_score_diff)
        @test isapprox(weight, expected_score_diff)

        new_tr2, weight2, _, rev2 = update(tr, ([1, 2],), (NoChange(),), spec, select(:world => :num_aircrafts => ()))
        @test isapprox(get_score(new_tr2), get_score(new_tr))
        @test isapprox(weight2, logpdf(poisson, 1, 5) - logpdf(poisson, 2, 5)) # only the constrained choices should impact the weight
        @test get_choices(new_tr2) == get_choices(new_tr)

        @test rev == rev2
        @test rev isa UpdateWithOUPMMovesSpec
        @test rev.moves == (BirthMove(Aircraft(1)),)

        @test rev.subspec == discarded_vals
    end

    sm_constraints = choicemap(
        (:world => :num_aircrafts => () => :num, 2),
        (:world => :num_blips => (Aircraft(1), Timestep(1)) => :num, 1),
        (:world => :num_blips => (Aircraft(2), Timestep(1)) => :num, 3),
        (:world => :num_blips => (Aircraft(1), Timestep(2)) => :num, 1),
        (:world => :num_blips => (Aircraft(2), Timestep(2)) => :num, 1),
    )

    AT(a::Int, t::Int) = (Aircraft(a), Timestep(t))
    Blp(a::Int, t::Int, i::Int) = Blip(AT(a, t), i)

    @testset "split moves" begin
        tr, _ = generate(get_blip_sizes_at_times, ([1, 2],), sm_constraints)

        # birth of object with no origins
        spec = UpdateWithOUPMMovesSpec(
            (
                SplitMove(
                    Blip((Aircraft(2), Timestep(1)), 2),
                    1, 2
                ),
            ),
            choicemap(
                (:world => :num_blips => (Aircraft(2), Timestep(1)) => :num, 4),
            )
        )
        new_tr, weight, _, rev = update(tr, ([1,2],), (NoChange(),), spec, invert(select(:world => :blip_size => Blip((Aircraft(2), Timestep(1)), 2))))

        old_dict = deepcopy(get_retval(tr))
        new_dict = get_retval(new_tr)

        @test new_dict[Blip((Aircraft(2), Timestep(1)), 3)] == old_dict[Blip((Aircraft(2), Timestep(1)), 1)]
        @test new_dict[Blip((Aircraft(2), Timestep(1)), 4)] == old_dict[Blip((Aircraft(2), Timestep(1)), 3)]
        asize = tr[:world => :plane_size => Aircraft(2)]

        expected_score_diff = (
            logpdf(poisson, 4, 1) - logpdf(poisson, 3, 1)
            + logpdf(normal, new_tr[:world => :blip_size => Blip((Aircraft(2), Timestep(1)), 1)], asize/600, 0.05)
            + logpdf(normal, new_tr[:world => :blip_size => Blip((Aircraft(2), Timestep(1)), 2)], asize/600, 0.05)
            - logpdf(normal, tr[:world => :blip_size => Blip((Aircraft(2), Timestep(1)), 2)], asize/600, 0.05)
        )
        @test isapprox(get_score(new_tr) - get_score(tr), expected_score_diff)
        @test isapprox(weight, logpdf(poisson, 4, 1) - logpdf(poisson, 3, 1))

        @test rev isa UpdateWithOUPMMovesSpec
        @test rev.moves == (MergeMove(Blip((Aircraft(2), Timestep(1)), 2), 1, 2),)
        @test rev.subspec == choicemap(
            (:world => :blip_size => Blip((Aircraft(2), Timestep(1)), 2) => :blip_size, tr[:world => :blip_size => Blip((Aircraft(2), Timestep(1)), 2)]),
            (:world => :num_blips => AT(2, 1) => :num, 3)
        )

        moves = (
            Blp(2, 1, 1) => Blp(1, 1, 2),
            Blp(2, 1, 2) => Blp(1, 1, 1),
            Blp(2, 1, 3) => Blp(3, 1, 1),
            Blp(2, 2, 1) => Blp(3, 2, 1)
        )
        spec = UpdateWithOUPMMovesSpec(
            (
                SplitMove(
                    Aircraft(2),
                    1, 3;
                    moves=moves
                ),
            ),
            choicemap(
                (:world => :num_aircrafts => () => :num, 3),
                (:world => :num_blips => AT(1, 1) => :num, 2),
                (:world => :num_blips => AT(3, 1) => :num, 1),
                (:world => :num_blips => AT(1, 2) => :num, 0),
                (:world => :num_blips => AT(3, 2) => :num, 1),
            )
        )
        new_tr, weight, _, rev = update(tr, ([1,2],), (NoChange(),), spec, AllSelection())

        @test new_tr[:world => :num_blips => AT(2, 1)] == tr[:world => :num_blips => AT(1, 1)]
        @test new_tr[:world => :num_blips => AT(2, 2)] == tr[:world => :num_blips => AT(1, 2)]

        new_dict = get_retval(new_tr)
        @test new_dict[Blp(2, 1, 1)] == old_dict[Blp(1, 1, 1)]
        @test new_dict[Blp(2, 2, 1)] == old_dict[Blp(1, 2, 1)]
        for (from, to) in moves
            @test old_dict[from] == new_dict[to]
        end

        newsize1 = new_tr[:world => :plane_size => Aircraft(1)]
        newsize3 = new_tr[:world => :plane_size => Aircraft(3)]
        oldsize2 = tr[:world => :plane_size => Aircraft(2)]
        new_bsize(b) = new_tr[:world => :blip_size => b]
        old_bsize(b) = tr[:world => :blip_size => b]
        new_planesize_scorediff = (
              logpdf(normal, newsize1, 600, 100)
            + logpdf(normal, newsize3, 600, 100)
            - logpdf(normal, tr[:world => :plane_size => Aircraft(2)], 600, 100)
        )
        new_blipsize_scoresum = (
            logpdf(normal, new_bsize(Blp(1, 1, 1)), newsize1/600, 0.05)
          + logpdf(normal, new_bsize(Blp(1, 1, 2)), newsize1/600, 0.05)
          + logpdf(normal, new_bsize(Blp(3, 1, 1)), newsize3/600, 0.05)
          + logpdf(normal, new_bsize(Blp(3, 2, 1)), newsize3/600, 0.05)
        )
        old_blipsize_scoresum = (
             logpdf(normal, old_bsize(Blp(2, 1, 1)), oldsize2/600, 0.05)
            + logpdf(normal, old_bsize(Blp(2, 1, 2)), oldsize2/600, 0.05)
            + logpdf(normal, old_bsize(Blp(2, 1, 3)), oldsize2/600, 0.05)
            + logpdf(normal, old_bsize(Blp(2, 2, 1)), oldsize2/600, 0.05)
        )
        blipsize_scorediff = new_blipsize_scoresum - old_blipsize_scoresum

        expected_score_diff = (
            logpdf(poisson, 3, 5) - logpdf(poisson, 2, 5)
            + logpdf(poisson, 2, 1) + logpdf(poisson, 1, 1) + logpdf(poisson, 0, 1) + logpdf(poisson, 1, 1)
            - logpdf(poisson, 3, 1) - logpdf(poisson, 1, 1)
            + new_planesize_scorediff
            + blipsize_scorediff
        )
        @test isapprox(get_score(new_tr) - get_score(tr), expected_score_diff)

        # weight should be the scorediff, minus the scores for the generated values
        @test isapprox(weight, expected_score_diff - logpdf(normal, newsize1, 600, 100) - logpdf(normal, newsize3, 600, 100))

        @test length(rev.moves) == 1
        @test rev.moves[1].to == Aircraft(2)
        @test rev.moves[1].from_idx_1 == 1
        @test rev.moves[1].from_idx_2 == 3
        @test Set(rev.moves[1].moves) == Set(map(((x, y),) -> y => x , moves))

        @test rev.subspec == choicemap(
            (:world => :num_aircrafts => () => :num, 2),
            (:world => :num_blips => AT(2, 1) => :num, 3),
            (:world => :num_blips => AT(2, 2) => :num, 1),
            (:world => :plane_size => Aircraft(2) => :size, oldsize2)
        )
    end

    @testset "merge moves" begin
        tr, _ = generate(get_blip_sizes_at_times, ([1, 2],), sm_constraints)

        oldbsize1 = tr[:world => :blip_size => Blp(2, 1, 1)]
        oldbsize3 = tr[:world => :blip_size => Blp(2, 1, 3)]
        meanbsize = (oldbsize1 + oldbsize3)/2.

        # merge of object with no origins
        spec = UpdateWithOUPMMovesSpec(
            (
                MergeMove(
                    Blip((Aircraft(2), Timestep(1)), 2),
                    1, 3
                ),
            ),
            choicemap(
                (:world => :num_blips => (Aircraft(2), Timestep(1)) => :num, 2),
                (:world => :blip_size => Blp(2, 1, 2) => :blip_size, meanbsize)
            )
        )
        new_tr, weight, _, rev = update(tr, ([1,2],), (NoChange(),), spec, AllSelection())

        old_dict = deepcopy(get_retval(tr))
        new_dict = deepcopy(get_retval(new_tr))

        @test old_dict[Blp(2, 1, 2)] == new_dict[Blp(2, 1, 1)]
        @test new_dict[Blp(2, 1, 2)] == meanbsize

        expected_weight = (
            logpdf(poisson, 2, 1) - logpdf(poisson, 3, 1)
            + logpdf(normal, meanbsize, tr[:world => :plane_size => Aircraft(2)]/600, 0.05)
            - logpdf(normal, oldbsize1, tr[:world => :plane_size => Aircraft(2)]/600, 0.05)
            - logpdf(normal, oldbsize3, tr[:world => :plane_size => Aircraft(2)]/600, 0.05)
        )

        @test isapprox(weight, expected_weight)
        @test isapprox(get_score(new_tr) - get_score(tr), expected_weight)

        @test rev.moves == (
            SplitMove(Blp(2, 1, 2), 1, 3),
        )

        @test rev.subspec == choicemap(
            (:world => :num_blips => AT(2, 1) => :num, 3),
            (:world => :blip_size => Blp(2, 1, 1) => :blip_size, oldbsize1),
            (:world => :blip_size => Blp(2, 1, 3) => :blip_size, oldbsize3)
        )

        # merge of object with origins
        oldsize1 = tr[:world => :plane_size => Aircraft(1)]
        oldsize2 = tr[:world => :plane_size => Aircraft(2)]
        meansize = (oldsize1 + oldsize2)/2
        moves = (
            Blp(1, 1, 1) => Blp(1, 1, 1),
            Blp(2, 1, 2) => Blp(1, 1, 2),
            Blp(2, 1, 1) => Blp(1, 1, 3),
            Blp(2, 1, 3) => Blp(1, 1, 4),
            Blp(1, 2, 1) => Blp(1, 2, 2),
            Blp(2, 2, 1) => Blp(1, 2, 1)
        )
        spec = UpdateWithOUPMMovesSpec(
            (
                MergeMove(
                    Aircraft(1),
                    1, 2;
                    moves=moves
                ),
            ),
            choicemap(
                (:world => :num_aircrafts => () => :num, 1),
                (:world => :num_blips => AT(1, 1) => :num, 4),
                (:world => :num_blips => AT(1, 2) => :num, 2),
                (:world => :plane_size => Aircraft(1) => :size, meansize)
            )
        )
        new_tr, weight, _, rev = update(tr, ([1,2],), (NoChange(),), spec, AllSelection())

        old_dict = get_retval(tr)
        new_dict = get_retval(new_tr)
        for (oldblip, newblip) in moves
            @test old_dict[oldblip] == new_dict[newblip]
        end

        new_bsize(b) = new_tr[:world => :blip_size => b]
        old_bsize(b) = tr[:world => :blip_size => b]
        new_planesize_scorediff = (
              logpdf(normal, meansize, 600, 100)
            - logpdf(normal, oldsize1, 600, 100)
            - logpdf(normal, oldsize2, 600, 100)
        )
        new_blipsize_scoresum = sum(
            logpdf(normal, new_bsize(b), meansize/600, 0.05)
            for (_, b) in moves
        )
        oldsize(x) = if x == 1; oldsize1; elseif x == 2; oldsize2; else; error(); end;
        old_blipsize_scoresum = sum(
            logpdf(normal, old_bsize(b), oldsize(b.origin[1].idx)/600, 0.05)
            for (b, _) in moves
        )
        blipsize_scorediff = new_blipsize_scoresum - old_blipsize_scoresum

        expected_score_diff = (
            logpdf(poisson, 1, 5) - logpdf(poisson, 2, 5)
            + logpdf(poisson, 4, 1) + logpdf(poisson, 2, 1)
            - logpdf(poisson, 3, 1) - 3*logpdf(poisson, 1, 1)
            + new_planesize_scorediff
            + blipsize_scorediff
        )
        @test isapprox(get_score(new_tr) - get_score(tr), expected_score_diff)
        @test isapprox(weight, expected_score_diff)

        @test rev.moves == (
            SplitMove(
                Aircraft(1),
                1, 2;
                moves=map(x -> (x.second => x.first), moves)
            ),
        )
        @test rev.subspec == choicemap(
            (:world => :num_aircrafts => () => :num, 2),
            (:world => :num_blips => AT(1, 1) => :num, 1),
            (:world => :num_blips => AT(1, 2) => :num, 1),
            (:world => :num_blips => AT(2, 1) => :num, 3),
            (:world => :num_blips => AT(2, 2) => :num, 1),
            (:world => :plane_size => Aircraft(1) => :size, oldsize1),
            (:world => :plane_size => Aircraft(2) => :size, oldsize2)
        )
    end
end