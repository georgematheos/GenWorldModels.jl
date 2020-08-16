verb_prior(rel) ~ Dirichlet(α)
sampled_fact(i) ~ UniformChoice({f for Fact F})
verb[i] ~ Categorical(
    verb_prior(relation(sampled_fact(i)))
)


Fact((Rel(A), Ent(Y), Ent(Y)), 1) --> Fact(L)

obs sampled_fact(2) = Fact(L)
verb[2] ~ Categorical(
    verb_prior(relation(Fact(L)))
)

verb[2] ~ Categorical(
    verb_prior(Rel(A))
)

verb[2] ~ Categorical(
    verb_prior(Rel(B))
)