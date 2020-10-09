struct Fact{R}
    rel::R
    ent1::Int
    ent2::Int
    Fact(rel::Int, ent1::Int, ent2::Int) = new{Int}(rel, ent1, ent2)
end
Base.isapprox(f1::Fact, f2::Fact) = f1.rel == f2.rel && f1.ent1 == f2.ent1 && f2.ent2 == f2.ent2

include("../model2/distributions.jl")
include("../model2/dirichlet_process_entity_mention.jl")
include("../model2/beta_bernoulli_subset.jl")

fact_getter(rel) = entpair -> Fact(rel, entpair...)
fact_getter(rel::Diffed{Int, NoChange}) = Diffed(fact_getter(strip_diff(rel)), NoChange())
fact_getter(rel::Diffed{Int, UnknownChange}) = Diffed(fact_getter(strip_diff(rel)), UnknownChange())
@gen (static, diffs) function generate_facts_for_rel(args)
    (rel, possible_entpairs, beta_prior) = args
    # b = println("possible entpairs: ", possible_entpairs)
    true_entpairs ~ beta_bernoulli_subset(possible_entpairs, beta_prior[1], beta_prior[2])
    # a = println("true entpairs: $true_entpairs")
    facts ~ no_collision_set_map(fact_getter(rel), true_entpairs)
    return facts
end

rel_to_args(possible_entpairs, beta_prior) = rel -> (rel, possible_entpairs, beta_prior)
function rel_to_args(p::Diffed{<:Any, NoChange}, b::Diffed{<:Any, NoChange})
    Diffed(rel_to_args(strip_diff(p), strip_diff(b)), NoChange())
end
function rel_to_args(p::Diffed, b::Diffed)
    Diffed(rel_to_args(strip_diff(p), strip_diff(b)), UnknownChange())
end
rangeset(l) = Set(1:l)
rangeset(l::Diffed{<:Any, UnknownChange}) = Diffed(rangeset(strip_diff(l)), UnknownChange())
@gen (static, diffs) function get_fact_set(num_entities, num_rels, beta_prior)
    possible_entpairs = [(i, j) for i=1:num_entities, j=1:num_entities]
    # r_to_args = Dict(r => rel_to_args(possible_entpairs, beta_prior)(r) for r in rangeset(num_rels))
    r_to_args = lazy_set_to_dict_map(rel_to_args(possible_entpairs, beta_prior), rangeset(num_rels))
    # display(r_to_args)
    facts_per_rel ~ DictMap(generate_facts_for_rel)(r_to_args)
    facts ~ tracked_union(facts_per_rel)
    #  x = println("All facts are facts: ", all(f isa Fact for f in facts))
    return facts
end

@gen (static, diffs) function sample_facts(num_entities, num_rels, beta_prior, num_sentences)
    all_facts ~ get_fact_set(num_entities, num_rels, beta_prior)
    sampled_facts ~ Map(uniform_choice)(fill(all_facts, num_sentences))
    # a = println("samples: $sampled_facts")
    return sampled_facts
end

@gen (static, diffs) function get_rel(fact); return fact.rel; end;
@gen (static, diffs) function get_entpair(fact); return (fact.ent1, fact.ent2); end;

@gen (static, diffs) function generate_sentences(
    num_entities,
    num_rels_prior,
    beta_prior,
    num_sentences,
    dirichlet_prior_val,
    num_verbs
)
    num_rels ~ discrete_log_normal(num_rels_prior[1], num_rels_prior[2])
    sampled_facts ~ sample_facts(num_entities, num_rels, beta_prior, num_sentences)
    rels ~ Map(get_rel)(sampled_facts)
    entpairs ~ Map(get_entpair)(sampled_facts)

    α = fill(dirichlet_prior_val, num_verbs)
    verbs ~ dirichlet_process_entity_mention(rels, α)

    return (verbs, entpairs)
end

@load_generated_functions()

model_args(p::ModelParams) = (p.num_entities, p.num_relations_prior, p.beta_prior, p.num_sentences, p.dirichlet_prior_val, p.num_verbs)
num_ents(tr) = get_args(tr)[1]
num_verbs(tr) = get_args(tr)[6]
dirichlet_prior_val(tr) = get_args(tr)[5]
entpairs(tr) = get_retval(tr)[2]
verbs(tr) = get_retval(tr)[1]