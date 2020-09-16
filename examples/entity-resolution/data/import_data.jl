struct LabeledDataSet
    sentences_numeric::Vector{SentenceNumeric}
    facts_numeric::Vector{FactNumeric}
    entity_strings::Vector{String}
    verb_strings::Vector{String}
    relation_strings::Vector{String}
end

read_data() = read_data("testdata.json")
function read_data(filename)
    data = JSON.parsefile(joinpath(@__DIR__, filename))

    sentences_numeric = []
    facts_numeric = []
    entity_strings = []
    verb_strings = []
    relation_strings = []

    verb_idx = Dict{String, Int}()
    ent_idx = Dict{String, Int}()
    get_verb_idx!(verb) = haskey(verb_idx, verb) ? verb_idx[verb] : begin
        push!(verb_strings, verb)
        verb_idx[verb] = length(verb_strings)
        length(verb_strings)
    end
    get_ent_idx!(ent) = haskey(ent_idx, ent) ? ent_idx[ent] : begin
        push!(entity_strings, ent)
        ent_idx[ent] = length(entity_strings)
        length(entity_strings)
    end

    for rel_datum in data["relData"]
        push!(relation_strings, rel_datum["relName"])
        rel_idx = length(relation_strings)
        for fact_datum in rel_datum["facts"]
            for sentence in fact_datum
                e1 = get_ent_idx!(sentence["source"])
                e2 = get_ent_idx!(sentence["dest"])
                verb = get_verb_idx!(sentence["depPath"])
                push!(facts_numeric, FactNumeric(rel=rel_idx, ent1=e1, ent2=e2))
                push!(sentences_numeric, SentenceNumeric(verb=verb, ent1=e1, ent2=e2))
            end
        end
    end

    return LabeledDataSet(sentences_numeric, facts_numeric, entity_strings, verb_strings, relation_strings)
end