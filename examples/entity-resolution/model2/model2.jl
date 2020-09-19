@type Relation
struct Fact{R}
    rel::R
    ent1::Int
    ent2::Int
    Fact(rel::R, ent1::Int, ent2::Int) where {R <: Relation} = new{R}(rel, ent1, ent2)
end
Base.isapprox(f1::Fact, f2::Fact) = f1.rel == f2.rel && f1.ent1 == f2.ent1 && f2.ent2 == f2.ent2

include("distributions.jl")
include("dirichlet_process_entity_mention.jl")
include("beta_bernoulli_subset.jl")

@gen (static, diffs) function num_relations(world, t)
    num_rels_prior ~ lookup_or_generate(world[:args][:num_rels_prior])
    num ~ discrete_log_normal(num_rels_prior[1], num_rels_prior[2])
    return num
end

fact_getter(rel) = entpair -> Fact(rel, entpair...)
fact_getter(rel::Diffed{Int, NoChange}) = Diffed(fact_getter(strip_diff(rel)), NoChange())
@gen (static, diffs) function generate_facts_for_rel(args)
    (rel, possible_entpairs, beta_prior) = args
    true_entpairs ~ beta_bernoulli_subset(possible_entpairs, beta_prior[1], beta_prior[2])
    facts ~ no_collision_set_map(fact_getter(rel), true_entpairs)
    return facts
end

rel_to_args(possible_entpairs, beta_prior) = rel -> (rel, possible_entpairs, beta_prior)
function rel_to_args(p::Diffed{<:Any, NoChange}, b::Diffed{<:Any, NoChange})
    Diffed(rel_to_args(strip_diff(p), strip_diff(b)), NoChange())
end
@gen (static, diffs) function get_fact_set(world)
    num_entities ~ lookup_or_generate(world[:args][:num_entities])
    possible_entpairs = [(i, j) for i=1:num_entities, j=1:num_entities]
    beta_prior ~ lookup_or_generate(world[:args][:beta_prior])

    rels ~ get_sibling_set(:Relation, :num_relations, world, ())

    rel_to_args_dict = lazy_set_to_dict_map(rel_to_args(possible_entpairs, beta_prior), rels)
    rels_to_facts ~ DictMap(generate_facts_for_rel)(world, rel_to_args_dict)
    facts ~ tracked_union(rels_to_facts)
    return facts
end

@gen (static, diffs) function sample_facts(world, num_sentences)
    all_facts ~ get_fact_set(world)
    sampled_facts ~ Map(uniform_fact_sample)(fill(world, num_sentences), fill(all_facts, num_sentences))
    return sampled_facts
end

@gen (static, diffs) function get_rel(fact); return fact.rel; end;
@gen (static, diffs) function get_entpair(fact); return (fact.ent1, fact.ent2); end;

@gen (static, diffs) function _generate_sentences(world, num_sentences, dirichlet_prior_val, num_verbs)
    sampled_facts ~ sample_facts(world, num_sentences)
    rels ~ Map(get_rel)(sampled_facts)
    entpairs ~ Map(get_entpair)(sampled_facts)
    
    α = fill(dirichlet_prior_val, num_verbs)
    verbs ~ dirichlet_process_entity_mention(rels, α)

    return (verbs, entpairs)
end

@load_generated_functions()

"""
    generate_sentences(num_entities, num_rels_prior, beta_prior, num_sentences, dirichlet_prior_val, num_verbs)

`beta_prior = (α, β)`, `num_rels_prior = (μ, σ)` (for the lognormal, so the mean is exp(μ + σ^2/2))
"""
generate_sentences = UsingWorld(_generate_sentences, :num_relations => num_relations;
    world_args=(:num_entities, :num_rels_prior, :beta_prior)
)

model_args(p::ModelParams) = (p.num_entities, p.num_relations_prior, p.beta_prior, p.num_sentences, p.dirichlet_prior_val, p.num_verbs)
num_ents(tr) = get_args(tr)[1]
num_verbs(tr) = get_args(tr)[6]
dirichlet_prior_val(tr) = get_args(tr)[5]
entpairs(tr) = get_retval(tr)[2]
verbs(tr) = get_retval(tr)[1]

function get_state(tr)
    nrels = tr[:world => :num_relations => ()]
    facts = tr[:kernel => :sampled_facts]
    rels_abstract = map(fact -> fact.rel, facts)
    rels_concrete = map(rel -> GenWorldModels.convert_to_concrete(tr.world, rel).idx, rels_abstract)
    facts_numeric = Set(
        FactNumeric(rel=GenWorldModels.convert_to_concrete(tr.world, fact.rel).idx, ent1=fact.ent1, ent2=fact.ent2)
        for fact in tr[:kernel => :sampled_facts => :all_facts]
    )
    State(nrels, facts_numeric, rels_concrete)
end