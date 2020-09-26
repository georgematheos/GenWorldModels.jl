using JSON

struct UnlabelledDataSet
    sentences_numeric::Vector{SentenceNumeric}
    entity_strings::Vector{String}
    verb_strings::Vector{String}
end

read_data(; num=nothing) = read_data("Umass-sub-corpus-06-12/pluieTriples_2013_06_12_3.json", num=num)
function read_data(filename; num=nothing)
    data = JSON.parsefile(joinpath(@__DIR__, filename))

    sentences_numeric = []
    entity_strings = []
    verb_strings = []

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

    sentences = num === nothing ? data["sentences"] : data["sentences"][1:num]

    for sentence in sentences
        e1 = get_ent_idx!(sentence["source"])
        e2 = get_ent_idx!(sentence["dest"])
        verb = get_verb_idx!(sentence["depPath"])
        push!(sentences_numeric, SentenceNumeric(verb=verb, ent1=e1, ent2=e2))
    end

    return UnlabelledDataSet(sentences_numeric, entity_strings, verb_strings)
end