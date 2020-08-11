NUM_WORDS = 5
α_val = 1.
α = fill(α_val, NUM_WORDS)

@type Object
@gen function num_objects(_, _) # memoized
    num ~ poisson(7)
    return num
end
@gen function prior_for_object(_, _::Object) # memoized
    prior ~ dirichlet(α)
    return prior
end
@gen function sample_mention(world::World, object::Object)
    prior ~ lookup_or_generate(world[:prior_for_object][object])
    mention ~ categorical(prior)
    return mention
end
@gen function _sample_entities(world, num_samples)
    object_set ~ get_object_set(world, Object)
    sampled_objs = [
        {:sample => i} ~ uniform_choice(object_set)
        for i=1:num_samples
    ]
    sampled_mentions = [
        {:mention => i} ~ sample_mention(world, obj)
        for obj in sampled_objs
    ]

    counts ~ get_emm_counts(sampled_objs, sampled_mentions)
    
    return sampled_mentions
end


@gen (static, diffs) function _sample_entities(world, num_samples)
    object_set ~ get_object_set(world, Object)
    sampled_objs ~ Map(uniform_choice)(fill(object_set, num_samples))
    sampled_mentions ~ Map(sample_mention)(fill(world, num_samples), sampled_objs)
    counts ~ get_emm_counts(sampled_objs, sampled_mentions)
    return sampled_mentions
end
sample_entities = UsingWorld(_sample_entities, :num_objects => num_objects, :prior_for_object => prior_for_object)

#=
Type Object;
#Object ~ poisson(7);
prior(Object o) ~ dirichlet(α);

Type Sample;
guaranteed Sample[num_samples];

obj(Sample s) ~ uniform_choice({o for Object o});
mention(Sample s) ~ categorical(prior(obj(s)));

Obs mention[Sample[1]] = ...;
Obs mention[Sample[2]] = ...;
...
=#



@gen (world) function sample_entities(num_samples)
    @type Object
    @mem @dist num_objects(::Tuple{}) = poisson(7)
    @num Object() = num_objects

    @mem @dist prior(::Object) = dirichlet(α)
    @gen function sample_mention(object::Object)
        prior ~ prior[object]
        mention ~ categorical(prior)
        return mention
    end

    object_set ~ get_object_set(Object)
    sampled_objs = [
        {:sample => i} ~ uniform_choice(object_set)
        for i=1:num_objects
    ]
    sampled_mentions = [
        {:mention => i} ~ sample_mention(obj)
        for obj in sampled_objs
    ]
    
    return sampled_mentions
end


@gen (static, diffs) function sample_objects_with_replacement(world)
    num_objs ~ lookup_or_generate(world[:num_objects][()])
    abstract_objs ~ Map(lookup_or_generate)(mgfcall_map(world[:abstract], 1:num_objs))
    sampled_objs ~ Map(uniform_from_set)(fill(Set(abstract_objs), num_samples))
    return sampled_objs
end
