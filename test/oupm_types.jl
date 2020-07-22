@type AudioSource

@gen (static) function get_volume(world::World, source::AudioSource{UUID})
    vol ~ normal(100, 4)
    idx ~ lookup_or_generate(world[:index][source])
    return idx => vol
end

@gen (static) function _vol_ind_kernel(world::World)
    num_sources ~ uniform_discrete(3, 7)
    num_samples ~ uniform_discrete(2, 3)
    samples ~ Map(uniform_discrete)(fill(1, num_samples), fill(num_sources, num_samples))
    sources = [AudioSource(idx) for idx in samples]
    idx_vol_pairs ~ Map(lookup_or_generate)([world[:volume][source] for source in sources])
    return idx_vol_pairs
end

@load_generated_functions()

get_vols_and_inds = UsingWorld(
    _vol_ind_kernel,
    :volume => get_volume;
    oupm_types=(AudioSource,)
)

@testset "open universe types" begin
@testset "type construction" begin
    @test AudioSource(1) isa AudioSource{Int}
    @test AudioSource(uuid4()) isa AudioSource{UUID}
    @test_throws MethodError AudioSource("i am a string")
    @test GenWorldModels.oupm_type(AudioSource(1)) == AudioSource
    @test GenWorldModels.oupm_type(AudioSource(uuid4())) == AudioSource
    @test AudioSource(Diffed(1, NoChange())) == Diffed(AudioSource(1), NoChange())
    @test AudioSource(Diffed(1, UnknownChange())) == Diffed(AudioSource(1), GenWorldModels.OUPMTypeIdxChange())
end

@testset "simple lookup_or_generate conversion" begin
    @gen function _simple_convert(world)
        src ~ lookup_or_generate(world[AudioSource][1])
        idx ~ lookup_or_generate(world[:index][src])
        return (idx, src)
    end
    simple_convert = UsingWorld(_simple_convert; oupm_types=(AudioSource,))

    tr = simulate(simple_convert, ())
    @test get_retval(tr)[1] == 1
    @test get_retval(tr)[2] isa AudioSource{UUID}
end
@testset "automatic conversion at lookup_or_generate for generate" begin
    tr = simulate(get_vols_and_inds, ())
    inds_to_vols = get_retval(tr)
    for ((out_index, _), in_index) in zip(inds_to_vols, tr[:kernel => :samples])
        @test out_index == in_index
    end
end

@testset "simple regenerate in `UsingWorld` with idx object in address" begin
    tr = simulate(get_vols_and_inds, ())
    idx = tr[:kernel => :samples][1]
    source = AudioSource(idx)
    
    @test !isempty(get_subtree(get_choices(tr), :world => :volume => source))
    
    acc = false
    new_tr = nothing
    while !acc # shouldn't need this; this should be a gibbs update.
        new_tr, acc = mh(tr, select(:world => :volume => source))
    end
    @test new_tr[:world => :volume => source] !== tr[:world => :volume => source]
    new_inds_to_vols = get_retval(new_tr)
    old_inds_to_vols = get_retval(tr)

    for (old, new) in zip(old_inds_to_vols, new_inds_to_vols)
        old_idx, old_vol = old
        new_idx, new_vol = new
        @test old_idx == new_idx
        if old_idx == idx
            @test old_vol != new_vol
        else
            @test old_vol == new_vol
        end
    end

    @test new_tr[:world => :volume => source] == (idx => get_choices(new_tr)[:world => :volume => source => :vol])

    # examine the internal `regenerate` call which should have occurred
    new_tr, weight, retdiff, discard = update(tr, get_args(tr), Tuple(NoChange() for arg in get_args(tr)), select(:world => :volume => source), EmptySelection())
    @test discard == choicemap((:world => :volume => source => :vol, tr[:world => :volume => source => :vol]))

    # P[new_tr]/P[old_tr] * Q[old_tr|selection]/Q[new_tr|selection] = 0 since here Q and P are the same since there are no downstream choices
    @test weight == 0.

    # should throw error if we don't specify that AudioSource is an OUPMType
    broken_get_vols_and_inds = UsingWorld(_vol_ind_kernel, :volume => get_volume)
    @test_throws Exception simulate(broken_get_vols_and_inds, ())

    # TODO: test updating with diffs enabled
end

@testset "idtable move_all_between" begin
    table = GenWorldModels.IDTable((AudioSource,))
    ids = Vector{UUID}(undef, 8)
    for i=1:8
        (table, id) = GenWorldModels.add_identifier_for(table, AudioSource, i)
        ids[i] = id
    end

    new_table, _ = GenWorldModels.move_all_between(table, GenWorldModels.oupm_type_name(AudioSource); min=2, max=5, inc=2)
    gid = idx -> GenWorldModels.get_id(new_table, AudioSource, idx)
    gidx = id -> GenWorldModels.get_idx(new_table, AudioSource, id)
    @test gid(1) == ids[1]
    @test gidx(ids[1]) == 1
    @test gid(4) == ids[2]
    @test gidx(ids[2]) == 4
    @test gid(5) == ids[3]
    @test gidx(ids[3]) == 5
    @test gid(6) == ids[4]
    @test gidx(ids[4]) == 6
    @test gid(7) == ids[5]
    @test gidx(ids[5]) == 7
    @test gid(8) == ids[8]
    @test gidx(ids[8]) == 8
    @test !GenWorldModels.has_id(new_table, AudioSource, ids[6])
    @test !GenWorldModels.has_id(new_table, AudioSource, ids[7])
    @test !GenWorldModels.has_idx(new_table, AudioSource, 2)
    @test !GenWorldModels.has_idx(new_table, AudioSource, 3)

    new_table, _ = GenWorldModels.move_all_between(table, GenWorldModels.oupm_type_name(AudioSource); min=3, max=6, inc=-2)
    gid = idx -> GenWorldModels.get_id(new_table, AudioSource, idx)
    gidx = id -> GenWorldModels.get_idx(new_table, AudioSource, id)
    @test !GenWorldModels.has_id(new_table, AudioSource, ids[1])
    @test !GenWorldModels.has_id(new_table, AudioSource, ids[2])
    @test !GenWorldModels.has_idx(new_table, AudioSource, 5)
    @test !GenWorldModels.has_idx(new_table, AudioSource, 6)
    @test gid(1) == ids[3]
    @test gidx(ids[3]) == 1
    @test gid(2) == ids[4]
    @test gidx(ids[4]) == 2
    @test gid(3) == ids[5]
    @test gidx(ids[5]) == 3
    @test gid(4) == ids[6]
    @test gidx(ids[6]) == 4
    @test gid(7) == ids[7]
    @test gidx(ids[7]) == 7
    @test gid(8) == ids[8]
    @test gidx(ids[8]) == 8

    # test what happens if there are gaps in the filled in indices
    table = GenWorldModels.IDTable((AudioSource,))
    (table, id2) = GenWorldModels.add_identifier_for(table, AudioSource, 2)
    (table, id3) = GenWorldModels.add_identifier_for(table, AudioSource, 3)
    (table, id5) = GenWorldModels.add_identifier_for(table, AudioSource, 5)
    new_table, _ = GenWorldModels.move_all_between(table, GenWorldModels.oupm_type_name(AudioSource); min=3, max=5, inc=-2)
    gid = idx -> GenWorldModels.get_id(new_table, AudioSource, idx)
    gidx = id -> GenWorldModels.get_idx(new_table, AudioSource, id)
    @test gid(3) == id5
    @test gidx(id5) == 3
    @test gid(1) == id3
    @test gidx(id3) == 1
    @test !GenWorldModels.has_id(new_table, AudioSource, id2)
    @test !GenWorldModels.has_idx(new_table, AudioSource, 2)
    @test !GenWorldModels.has_idx(new_table, AudioSource, 4)
    @test !GenWorldModels.has_idx(new_table, AudioSource, 5)

    new_table, _ = GenWorldModels.move_all_between(table, GenWorldModels.oupm_type_name(AudioSource); min=3, max=5, inc=-1)
    gid = idx -> GenWorldModels.get_id(new_table, AudioSource, idx)
    gidx = id -> GenWorldModels.get_idx(new_table, AudioSource, id)
    @test gid(2) == id3
    @test gidx(id3) == 2
    @test gid(4) == id5
    @test gidx(id5) == 4
    @test !GenWorldModels.has_id(new_table, AudioSource, id2)
    @test !GenWorldModels.has_idx(new_table, AudioSource, 1)
    @test !GenWorldModels.has_idx(new_table, AudioSource, 3)
    @test !GenWorldModels.has_idx(new_table, AudioSource, 5)

    new_table, _ = GenWorldModels.move_all_between(table, GenWorldModels.oupm_type_name(AudioSource); min=3, max=5, inc=+2)
    gid = idx -> GenWorldModels.get_id(new_table, AudioSource, idx)
    gidx = id -> GenWorldModels.get_idx(new_table, AudioSource, id)
    @test gid(2) == id2
    @test gidx(id2) == 2
    @test gid(5) == id3
    @test gidx(id3) == 5
    @test gid(7) == id5
    @test gidx(id5) == 7
    @test !GenWorldModels.has_idx(new_table, AudioSource, 1)
    @test !GenWorldModels.has_idx(new_table, AudioSource, 3)
    @test !GenWorldModels.has_idx(new_table, AudioSource, 4)
    @test !GenWorldModels.has_idx(new_table, AudioSource, 6)
end

function test_audiosource_moves_successful(old_tr, new_tr; expected_moves, expected_new)
    oldch = get_choices(old_tr)
    newch = get_choices(new_tr)
    for (old, new) in expected_moves
        if !(get_subtree(oldch, :world => :volume => AudioSource(old)) == get_subtree(newch, :world => :volume => AudioSource(new)))
            println("old=$old, new=$new")
        end
        @test get_subtree(oldch, :world => :volume => AudioSource(old)) == get_subtree(newch, :world => :volume => AudioSource(new))
    end
    for newidx in expected_new
        # check that every index where we should have a new value 
        for oldidx in (2, 3, 5)
            @test oldch[:world => :volume => AudioSource(oldidx) => :vol] != newch[:world => :volume => AudioSource(newidx) => :vol]
        end
    end
end
function test_correct_reverse_move(reverse_spec; expected_move, expected_constrained)
    @test reverse_spec isa UpdateWithOUPMMovesSpec
    @test reverse_spec.moves == (expected_move,)
    @test length(collect(get_subtrees_shallow(get_subtree(reverse_spec.subspec, :world => :volume)))) == length(expected_constrained)
    for idx in expected_constrained
        @test !isempty(get_subtree(reverse_spec.subspec, :world => :volume => AudioSource(idx)))
    end
end

@testset "oupm moves - basics" begin
    constraints_2_samples = choicemap(
        (:kernel => :num_sources, 5),
        (:kernel => :num_samples, 2),
        (:kernel => :samples => 1, 2),
        (:kernel => :samples => 2, 3)
    )

    @testset "simple birth moves" begin
        tr, weight = generate(get_vols_and_inds, (), constraints_2_samples)
        @test isapprox(weight, -(log(5) + log(2) + 2*log(5)))
        ch = get_choices(tr)

        # birth a new object at index 3.  Since we are looking up index 3 (but not index 4), this means the new index 3 will be
        # generated newly, and we discard the old index 3 (which is now at index 4 and is not looked up).
        spec = UpdateWithOUPMMovesSpec((BirthMove(AudioSource, 3),), choicemap((:kernel => :num_sources, 6)))
        new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
        newch = get_choices(new_tr)
        s2 = AudioSource(2); s3 = AudioSource(3)
        @test length(collect(get_subtrees_shallow(get_subtree(ch, :world => :volume)))) == 2
        @test get_subtree(newch, :world => :volume => s2) == get_subtree(ch, :world => :volume => s2)
        @test length(collect(get_subtrees_shallow(get_subtree(newch, :world => :volume)))) == 2
        @test new_tr[:world => :volume => s3] != tr[:world => :volume => s3] # should have been regenerated
        # generating weight not included since this is cancelled out by the Q distribution
        expected_weight = -logpdf(normal, tr[:world => :volume => s3 => :vol], 100, 4) + 2*(log(1/6) - log(1/5))
        @test isapprox(expected_weight, weight)
        @test new_tr[][1] == tr[][1]
        @test new_tr[][2] != tr[][2]

        @test reverse_move isa UpdateWithOUPMMovesSpec
        @test reverse_move.moves == (DeathMove(AudioSource, 3),)
        expected_reverse_constraints = choicemap((:kernel => :num_sources, 5))
        set_submap!(expected_reverse_constraints, :world => :volume => s3, get_submap(ch, :world => :volume => s3))
        @test reverse_move.subspec == expected_reverse_constraints

        # birth a new object at index 2.  Should generate new trace at index 2, move current trace at 2 to 3, and discard current 3.
        spec = UpdateWithOUPMMovesSpec((BirthMove(AudioSource, 2),), choicemap((:kernel => :num_sources, 6)))
        new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
        newch = get_choices(new_tr)
        @test get_subtree(newch, :world => :volume => s3) == get_subtree(ch, :world => :volume => s2)
        @test length(collect(get_subtrees_shallow(get_subtree(newch, :world => :volume)))) == 2
        @test new_tr[:world => :volume => s2] != tr[:world => :volume => s2] # should have been regenerated
        expected_weight = -logpdf(normal, tr[:world => :volume => s3 => :vol], 100, 4) + 2*(log(1/6) - log(1/5))
        @test isapprox(expected_weight, weight)
        @test new_tr[][2].second == tr[][1].second
        @test new_tr[][1] != tr[][1]

        @test reverse_move isa UpdateWithOUPMMovesSpec
        @test reverse_move.moves == (DeathMove(AudioSource, 2),)
        expected_reverse_constraints = choicemap((:kernel => :num_sources, 5))
        set_submap!(expected_reverse_constraints, :world => :volume => s3, get_submap(ch, :world => :volume => s3))
        @test reverse_move.subspec == expected_reverse_constraints
   end

    @testset "simple death moves" begin
        tr, weight = generate(get_vols_and_inds, (), constraints_2_samples)
        @test isapprox(weight, -(log(5) + log(2) + 2*log(5)))
        ch = get_choices(tr)

        spec = UpdateWithOUPMMovesSpec((DeathMove(AudioSource, 3),), choicemap((:kernel => :num_sources, 4)))
        new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
        newch = get_choices(new_tr)
        s2 = AudioSource(2); s3 = AudioSource(3)
        @test get_subtree(newch, :world => :volume => s2) == get_subtree(ch, :world => :volume => s2)
        @test length(collect(get_subtrees_shallow(get_subtree(newch, :world => :volume)))) == 2
        @test new_tr[:world => :volume => s3] != tr[:world => :volume => s3] # should have been regenerated
        # generating weight not included since this is cancelled out by the Q distribution
        expected_weight = -logpdf(normal, tr[:world => :volume => s3 => :vol], 100, 4) + 2*(log(1/4) - log(1/5))
        @test isapprox(expected_weight, weight)
        @test new_tr[][1] == tr[][1]
        @test new_tr[][2] != tr[][2]    

        @test reverse_move isa UpdateWithOUPMMovesSpec
        @test reverse_move.moves == (BirthMove(AudioSource, 3),)
        expected_reverse_constraints = choicemap((:kernel => :num_sources, 5))
        set_submap!(expected_reverse_constraints, :world => :volume => s3, get_submap(ch, :world => :volume => s3))
        @test reverse_move.subspec == expected_reverse_constraints

        # delete 2, so 3 moves to 2, and a new 3 is generated
        spec = UpdateWithOUPMMovesSpec((DeathMove(AudioSource, 2),), choicemap((:kernel => :num_sources, 4)))
        new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
        newch = get_choices(new_tr)
        @test get_subtree(newch, :world => :volume => s2) == get_subtree(ch, :world => :volume => s3)
        @test length(collect(get_subtrees_shallow(get_subtree(newch, :world => :volume)))) == 2
        @test new_tr[:world => :volume => s3] != tr[:world => :volume => s3] # should have been regenerated
        expected_weight = -logpdf(normal, tr[:world => :volume => s2 => :vol], 100, 4) + 2*(log(1/4) - log(1/5))
        @test isapprox(expected_weight, weight)
        @test new_tr[][2] !== tr[][2]
        @test new_tr[][1].second == tr[][2].second
        @test new_tr[][1] != tr[][1]

        @test reverse_move isa UpdateWithOUPMMovesSpec
        @test reverse_move.moves == (BirthMove(AudioSource, 2),)
        expected_reverse_constraints = choicemap((:kernel => :num_sources, 5))
        set_submap!(expected_reverse_constraints, :world => :volume => s2, get_submap(ch, :world => :volume => s2))
        @test reverse_move.subspec == expected_reverse_constraints
    end

    constraints_3_samples = choicemap(
        (:kernel => :num_sources, 5),
        (:kernel => :num_samples, 3),
        (:kernel => :samples => 1, 2),
        (:kernel => :samples => 2, 3),
        (:kernel => :samples => 3, 5)
    )

    @testset "simple split moves" begin
        tr, weight = generate(get_vols_and_inds, (), constraints_3_samples)

        spec = UpdateWithOUPMMovesSpec((SplitMove(AudioSource, 2, 4, 6),), choicemap((:kernel => :num_sources, 6)))
        new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
        test_audiosource_moves_successful(tr, new_tr; expected_moves=(3=>2,5=>5), expected_new=(3,))
        test_correct_reverse_move(reverse_move; expected_move=MergeMove(AudioSource, 2, 4, 6), expected_constrained=(2,))

        spec = UpdateWithOUPMMovesSpec((SplitMove(AudioSource, 4, 2, 5),), choicemap((:kernel => :num_sources, 6)))
        new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
        test_audiosource_moves_successful(tr, new_tr; expected_moves=(2 => 3,), expected_new=(2, 5))
        test_correct_reverse_move(reverse_move; expected_move=MergeMove(AudioSource, 4, 2, 5), expected_constrained=(3, 5))
        
        spec = UpdateWithOUPMMovesSpec((SplitMove(AudioSource, 5, 4, 2),), choicemap((:kernel => :num_sources, 6)))
        new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
        test_audiosource_moves_successful(tr, new_tr; expected_moves=(2 => 3, 3=>5), expected_new=(2,))
        test_correct_reverse_move(reverse_move; expected_move=MergeMove(AudioSource, 5, 4, 2), expected_constrained=(5,))
    end

    @testset "simple merge moves" begin
        tr, weight = generate(get_vols_and_inds, (), constraints_3_samples)

        spec = UpdateWithOUPMMovesSpec((MergeMove(AudioSource, 2, 4, 5),), choicemap((:kernel => :num_sources, 4), (:kernel => :samples => 3, 4)))
        new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
        test_audiosource_moves_successful(tr, new_tr; expected_moves=(2=>3,3=>4), expected_new=(2,))
        test_correct_reverse_move(reverse_move; expected_move=SplitMove(AudioSource, 2, 4, 5), expected_constrained=(5,))

        spec = UpdateWithOUPMMovesSpec((MergeMove(AudioSource, 4, 2, 5),), choicemap((:kernel => :num_sources, 4), (:kernel => :samples => 3, 4)))
        new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
        test_audiosource_moves_successful(tr, new_tr; expected_moves=(3=>2,), expected_new=(3, 4))
        test_correct_reverse_move(reverse_move; expected_move=SplitMove(AudioSource, 4, 2, 5), expected_constrained=(2, 5))
        
        spec = UpdateWithOUPMMovesSpec((MergeMove(AudioSource, 4, 3, 2),), choicemap((:kernel => :num_sources, 4), (:kernel => :samples => 3, 4)))
        new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
        test_audiosource_moves_successful(tr, new_tr; expected_moves=(5=>3,), expected_new=(2,4))
        test_correct_reverse_move(reverse_move; expected_move=SplitMove(AudioSource, 4, 3, 2), expected_constrained=(2,3))
    end

    @testset "simple move moves" begin
        tr, weight = generate(get_vols_and_inds, (), constraints_3_samples)
        spec = UpdateWithOUPMMovesSpec((MoveMove(AudioSource, 2, 5),), EmptyChoiceMap())
        new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
        test_audiosource_moves_successful(tr, new_tr; expected_moves=(3=>2,2=>5), expected_new=(3,))
        test_correct_reverse_move(reverse_move; expected_move=MoveMove(AudioSource, 5, 2), expected_constrained=(5,))

        spec = UpdateWithOUPMMovesSpec((MoveMove(AudioSource, 4, 2),), EmptyChoiceMap())
        new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
        test_audiosource_moves_successful(tr, new_tr; expected_moves=(2=>3,5=>5), expected_new=(2,))
        test_correct_reverse_move(reverse_move; expected_move=MoveMove(AudioSource, 2, 4), expected_constrained=(3,))

        spec = UpdateWithOUPMMovesSpec((MoveMove(AudioSource, 5, 2),), EmptyChoiceMap())
        new_tr, weight, _, reverse_move = update(tr, (), (), spec, AllSelection())
        test_audiosource_moves_successful(tr, new_tr; expected_moves=(2=>3,5=>2), expected_new=(5,))
        test_correct_reverse_move(reverse_move; expected_move=MoveMove(AudioSource, 2, 5), expected_constrained=(3,))
    end
end


end