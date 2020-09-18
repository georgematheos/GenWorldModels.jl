using Parameters: @with_kw

@with_kw struct SentenceNumeric
    verb::Int
    ent1::Int
    ent2::Int
end
@with_kw struct FactNumeric
    rel::Int
    ent1::Int
    ent2::Int
end

struct State
    num_relations::Int
    facts::AbstractSet{FactNumeric}
    sentence_rels::AbstractVector{Int}
end

function write_state!(io, state)
    println(io, state.num_relations)
    for fact in state.facts
        print(io, fact.rel, " ", fact.ent1, " ", fact.ent2, ",")
    end
    println(io, "")
    for rel in state.sentence_rels
        print(io, rel, ",")
    end
end
function read_state(io)
    num_rels = parse(Int, readline(io))

    factline = readline(io)
    factstrs = split(factline, ",")[1:end-1]
    split_factstrs = [split(factstr, " ") for factstr in factstrs]
    facts = Set(FactNumeric(
        rel=parse(Int, factstr[1]),
        ent1=parse(Int, factstr[2]),
        ent2=parse(Int, factstr[3])
    ) for factstr in split_factstrs)

    relline = readline(io)
    relstrs = split(relline, ",")[1:end-1]
    rels = [parse(Int, relstr) for relstr in relstrs]

    State(num_rels, facts, rels)
end

@with_kw struct ModelParams
    num_entities::Int
    num_verbs::Int
    num_sentences::Int
    dirichlet_prior_val::Float64
    beta_prior::Tuple{Float64, Float64}
    num_relations_prior::Tuple{Float64, Float64}
end
function ModelParams(;num_entities, num_verbs, num_sentences, dirichlet_prior_val, beta_prior, num_relations_mean, num_relations_var)
    num_relations_mean += 0.5 # approximately account for the `floor` in the discrete_log_normal
    σ2 = log(1 + num_relations_var/num_relations_mean^2)
    μ = log(num_relations_mean) - σ2/2
    ModelParams(num_entities, num_verbs, num_sentences, dirichlet_prior_val, beta_prior, (μ, √(σ2)))
end