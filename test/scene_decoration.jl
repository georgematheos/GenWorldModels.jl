struct Object
    size::Float64
    color::Int
end
Object(sz::Diffed, color) = Diffed(Object(strip_diff(sz), color), UnknownChange())
Object(sz, color::Diffed) = Diffed(Object(sz, strip_diff(color)), UnknownChange())
Object(sz::Diffed, color::Diffed) = Diffed(Object(strip_diff(sz), strip_diff(color)), UnknownChange())

@gen (static, diffs) function generate_object(world, idx)
    size ~ normal(idx, 1)
    color ~ uniform_discrete(1, 6)
    obj = Object(size, color)
    return obj
end

@gen (static, diffs) function sample_object(world, num_objects)
    idx ~ uniform_discrete(1, num_objects)
    obj ~ lookup_or_generate(world[:objects][idx])
    return obj
end
sample_objects = Map(sample_object)

@gen (static, diffs) function generate_scene_kernel(world)
    num_objects ~ poisson(30)
    num_in_scene ~ poisson(3)
    scene ~ sample_objects(fill(world, num_in_scene), fill(num_objects, num_in_scene))
    return scene
end
generate_scene = UsingWorld(generate_scene_kernel, :objects => generate_object)
@load_generated_functions()

@testset "scene decoration via UsingWorld" begin
    tr, weight = generate(generate_scene, (), choicemap(
        (:kernel => :num_in_scene, 3),
        (:kernel => :num_objects, 30),
        (:kernel => :scene => 1 => :idx, 1),
        (:kernel => :scene => 2 => :idx, 2),
        (:kernel => :scene => 3 => :idx, 3)
    ))
    
    @test get_retval(tr)[1] isa Object
    @test get_retval(tr)[1].color == tr[:world => :objects => 1 => :color]

    # weight should P[all_choices] - P[unconstrained_choices] = P[constrained choices]
    @test weight ≈ logpdf(poisson, 3, 3) + logpdf(poisson, 30, 30) + 3*log(1/30)

    # this chould account for all the unconstrained choices:
    obj_gen_scores = [log(1/6) + logpdf(normal, tr[:world => :objects => i => :size], i, 1) for i=1:3]

    # P[all_choices] = P[unconstrained_choices] + P[constrained_choices]
    @test get_score(tr) ≈ weight + sum(obj_gen_scores)

    # delete the last index on Map call:
    new_tr, weight, retdiff, discard = update(tr, (), (), choicemap((:kernel => :num_in_scene, 2)))
    # should mean we no longer have a lookup for object 3:
    @test get_submap(get_choices(new_tr), :world => :objects => 3) == EmptyChoiceMap()
    @test weight ≈ logpdf(poisson, 2, 3) - logpdf(poisson, 3, 3) - log(1/30) - (log(1/6) + logpdf(normal, tr[:world => :objects => 3 => :size], 3, 1))
    @test get_score(new_tr) - get_score(tr) ≈ weight

    new_color = tr[:world => :objects => 1 => :color] == 1 ? 2 : 1 # pick a new color for the first object
    new_tr, weight, retdiff, discard = update(tr, (), (), choicemap((:world => :objects => 1 => :color, new_color)))

    @test get_retval(new_tr)[1].color == new_color
    # all colors are equally likely so this shouldn't change any probabilities
    @test get_score(new_tr) == get_score(tr)
    @test weight == 0
end