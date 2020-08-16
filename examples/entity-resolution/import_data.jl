
function read_fact_data()
    fact_data = JSON.parsefile(joinpath(@__DIR__, "testdata.json"))
    fact_data = fact_data["facts"]
    idx = 1
    sentences = Tuple{Int, Int, Int}[]
    entities = Dict()
    verbs = Dict()
    ent_idx = 1
    verb_idx = 1
    fact_idx=1
    fact_idx_to_sentence_indices = []
    sentence_idx_to_fact = []
    for (sentences_for_fact) in fact_data
        sentence_indices_for_fact = []
        for sentence in sentences_for_fact
            ent1 = sentence["source"]
            ent2 = sentence["dest"]
            verb = sentence["depPath"]
            if !haskey(entities, ent1)
                entities[ent1] = ent_idx
                ent_idx += 1
            end
            if !haskey(entities, ent2)
                entities[ent2] = ent_idx
                ent_idx += 1
            end
            if !haskey(verbs, verb)
                verbs[verb] = verb_idx
                verb_idx += 1
            end
            push!(sentences, (entities[ent1], verbs[verb], entities[ent2]))
            push!(sentence_idx_to_fact, fact_idx)
            push!(sentence_indices_for_fact, idx)
            idx += 1
        end
        fact_idx += 1
        push!(fact_idx_to_sentence_indices, sentence_indices_for_fact)
    end

    return (sentences, collect(enumerate(fact_idx_to_sentence_indices)), sentence_idx_to_fact, length(entities), length(verbs))
end