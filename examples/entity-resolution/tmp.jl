@gen function generate_sentences(num_sentences, α, β, dirichlet_prior, num_entities)

    sentences ~ generate_sentences_given_facts(num_sentences, num_relations, num_entities, true_facts, dirichlet_prior)
end

@gen function generate_rels_and_true_facts(α, β, num_entities)
    num_relations ~ poisson(7)
    true_facts = Set()
    for rel=1:num_relations
        sparsity = {:sparsity => rel} ~ beta(α, β)
        for e1=1:num_entities, e2=1:num_entities
            is_true = {:is_true => (rel, e1, e2)} ~ bernoulli(sparsity)
            if is_true
                push!(true_facts, Fact((rel, e1, e2), 1))
            end
        end
    end
    return (num_relations, true_facts)
end


@gen (static) function observe_noisy_sum()
    val1 ~ normal(0, 1)
    val2 ~ normal(0, 2)
    sum = val1 + val2
    obs ~ normal(sum, 1)
    return obs
end

@gen (static) function get_vector_of_values()
    num_vals ~ poisson(5)
    # same as vals ~ [normal(0, 1) for _=1:num_vals]
    vals ~ Map(normal)(fill(0, num_vals), fill(1, num_vals))
    return vals
end



@birth(Relation(3))

# implemented as
@write(new_tr[:num_relations], old_num_rels + 1)
for rel=3:num_relations
    @copy(from=(:sparsity => rel) to=(:sparsity => rel+1))
    for e1=1:num_entities, e2=1:num_entities
        @copy(
            from=(:is_true => (rel, e1, e2)),
            to=(:is_true => (rel+1, e1, e2))
        )
    end
end


@gen function generate_rels_and_true_facts(concrete_to_abstract, α, β, num_entities)
    num_relations ~ poisson(7)
    true_facts = Set()
    for rel=1:num_relations
        abstract_rel = lookup_or_generate_abstract_form!(concrete_to_abstract, rel)

        sparsity = {:sparsity => abstract_rel} ~ beta(α, β)
        for e1=1:num_entities, e2=1:num_entities
            is_true = {:is_true => (abstract_rel, e1, e2)} ~ bernoulli(sparsity)
            if is_true
                push!(true_facts, Fact((abstract_rel, e1, e2), 1))
            end
        end
    end
    return (num_relations, true_facts)
end

@birth(Relation(3))

# implemented via:
@write(new_tr[:num_relations], old_num_rels + 1)

for rel=old_num_rels:-1:3
    concrete_to_abstract[rel + 1] = concrete_to_abstract[rel]
end
concrete_to_abstract[3] = generate_unique_identifier()

# split Relation(1) into 2 relations indexed Relation(1) and Relation(2)
@split(Rel(1), 1, 2,
    moves=(
        Fact((Rel(1), Ent(2), Ent(2)), 1) => Fact((Rel(1), Ent(2), Ent(2)), 1),
        Fact((Rel(1), Ent(1), Ent(2)), 1) => Fact((Rel(2), Ent(1), Ent(2)), 1)
    )
)

FactTriple(
    (
        Fact((Rel(1), Ent(1), Ent(2)), 1),
        Fact((Rel(4), Ent(2), Ent(2)), 1),
        Fact((Rel(1), Ent(2), Ent(2)), 1),
    ),
    1
)

Rel(1) --> Rel(A)
Ent(1) --> Ent(X)
Ent(2) --> Ent(Y)
Fact((Rel(A), Ent(Y), Ent(Y)), 1) --> Fact(L)
Fact((Rel(A), Ent(X), Ent(Y)), 1) --> Fact(M)


Rel(1) --> Rel(B)
Rel(2) --> Rel(C)
Ent(1) --> Ent(X)
Ent(2) --> Ent(Y)
Fact((Rel(B), Ent(Y), Ent(Y)), 1) --> Fact(L)
Fact((Rel(C), Ent(X), Ent(Y)), 1) --> Fact(M)

@gen function sample_sentence_for_fact(fact::AbstractObject{:Fact})
    (rel, _, _) = origin(fact)
    verb_prior = verb_prior(rel)

end

# trace a call to a distribution/generative function:
x ~ normal(0, 1)
y ~ generate_new_value(x)

# look-up a memoized, stochastic property, from the "world":
verb_prior ~ @w verb_prior[relation]

@gen function sample_verb_for_fact(world, fact)
    origin ~ @w origin[fact]
    (rel, _, _) = origin
    verb_prior ~ @w verb_prior[rel]
    verb ~ categorical(verb_prior)
    return verb
end

@gen (mem) function verb_prior(world, relation)
    α ~ @w α_for_relation[relation]
    verb_prior ~ dirichlet(α)
    return verb_prior
end

@gen function sample_verb_for_fact(world, fact)
    origin ~ @w origin[fact]
    (rel, _, _) = origin
    verb_prior ~ @w verb_prior[fact]
    verb ~ categorical(verb_prior)
    return verb
end

@gen function generate_verbs(world, list_of_facts)
    verbs = []
    for fact in list_of_facts
        verb ~ sample_verb_for_fact(world, fact)
        push!(verbs, verb)
    end
    return verbs
end

# we need to update α, then the verb prior, then the sampled verb
@write(new_tr[:world => :α_for_relation => Rel(5)], new_α)
@write(new_tr[:world => :verb_prior => Rel(5)], new_verb_prior)
@write(new_tr[:kernel => :sample_verb_for_fact => 1], new_verb)