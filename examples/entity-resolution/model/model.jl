include("distributions.jl")
include("uniform_choice.jl")
include("emm_counts.jl")

@type Entity
@type Relation
@type Fact

@gen (static#=, diffs=#) function dirichlet_prior(world, o::Tuple{}) # memoized; deterministic
    num_verbs ~ lookup_or_generate(world[:args][:num_verbs])
    return fill(DIRICHLET_PRIOR_VAL, num_verbs)
end

@gen (static#=, diffs=#) function generate_verb_prior(world, rel::Relation) # memoized
    dirichlet_prior ~ lookup_or_generate(world[:dirichlet_prior][()])
    prior ~ dirichlet(dirichlet_prior)
    return prior
end

@gen (static#=, diffs=#) function sample_verb(world, rel::Relation) # nonmemoized
    prior ~ lookup_or_generate(world[:verb_prior][rel])
    verb ~ categorical(prior)
    return verb
end

@gen (static#=, diffs=#) function generate_sentence(world, fact::Fact) # nonmemoized
    origin ~ lookup_or_generate(world[:origin][fact])
    (rel, ent1, ent2) = origin
    ent1_idx ~ lookup_or_generate(world[:index][ent1])
    ent2_idx ~ lookup_or_generate(world[:index][ent2])
    verb ~ sample_verb(world, rel)
    return (rel, (ent1_idx, verb, ent2_idx))
end

@gen (static#=, diffs=#) function num_relations(world, o::Tuple{}) # memoized
    num ~ poisson(NUM_REL_PRIOR_MEAN)
    return num
end
@gen (static#=, diffs=#) function sparsity(world, rel::Relation) # memoized
    sparsity ~ beta(BETA_PRIOR[1], BETA_PRIOR[2])
    return sparsity
end

@gen (static#=, diffs=#) function num_facts(world, origin)
    (rel, ent1, ent2) = origin
    truth_prior ~ lookup_or_generate(world[:sparsity][rel])
    is_true ~ bernoulli(truth_prior)
    return is_true
end

@gen  (static) function get_facts_for_origin(world, origin::Tuple{<:Relation, <:Entity, <:Entity}) # memoized
    is_true ~ lookup_or_generate(world[:num_facts][origin])
    return is_true ? [Fact(origin, 1)] : []
end

@gen (static) function get_facts_for_relation(world, relation)
    num_entities ~ lookup_or_generate(world[:args][:num_entities])
    origins = [(relation, Entity(e1), Entity(e2)) for e1=1:num_entities, e2=1:num_entities]
    facts ~ Map(get_facts_for_origin)(fill(world, num_entities^2), origins)
    flattened = collect(Iterators.flatten(facts))
    abstract_facts ~ Map(lookup_or_generate)(mgfcall_map(world[:abstract], flattened))
    return abstract_facts
end

@gen (static) function get_fact_set(world) #nonmemoized
    num_relations ~ lookup_or_generate(world[:num_relations][()])
    facts_per_rel ~ Map(get_facts_for_relation)(fill(world, num_relations), [Relation(i) for i=1:num_relations])
    factset = Set(collect(Iterators.flatten(facts_per_rel))) # TODO: unpack this in a way that is efficient w.r.t. diffs
    return factset
end

@gen (static#=, diffs=#) function sample_facts(world, num_sentences) # nonmemoized
    facts ~ get_fact_set(world)
    sampled_facts ~ Map(uniform_choice)(fill(world, num_sentences), fill(facts, num_sentences))
    return sampled_facts
end

@gen (static#=, diffs=#) function _generate_sentences(world, num_sentences) # kernel
    facts ~ sample_facts(world, num_sentences)
    rels_and_sentences ~ Map(generate_sentence)(fill(world, num_sentences), facts)
    
    num_verbs ~ lookup_or_generate(world[:args][:num_verbs])
    rels = map((rel, _) -> rel, rels_and_sentences)
    verbs = map((rel, (_, verb, _)) -> verb, rels_and_sentences)
    counts ~ get_entity_mention_counts(num_verbs, rels, verbs)

    sentences = map((_, sent) -> sent, rels_and_sentences)

    return sentences
end
@load_generated_functions()
generate_sentences = UsingWorld(_generate_sentences,
    :dirichlet_prior => dirichlet_prior,    
    :verb_prior => generate_verb_prior,
    :num_relations => num_relations,
    :sparsity => sparsity,
    :num_facts => num_facts
    ;
    world_args=(:num_entities, :num_verbs)#,
    # nums=(
    #     (Relation, (), :world => :num_relations => () => :num),
    #     (Entity, (), :world => :args => :num_entities),
    #     (Fact, (Relation, Entity, Entity), :world => 
    # )
)

# called via:
# generate_sentences(num_entities, num_verbs, num_sentences_to_generate)


#=

=#
#=

@gen (static#=, diffs=#) function get_fact_set(world)
    entset ~ GetOriginlessSet(:Entity, :args => :num_entities)(world)
    relset ~ GetOriginlessSet(:Relation, :num_relations => ())(world)
    fact_set ~ GetObjectSetFromOriginSets(:Fact)(world, :num_facts, relset, entset, entset)
    return fact_set
end

=#