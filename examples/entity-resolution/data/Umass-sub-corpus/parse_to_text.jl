using JSON

struct ORLMTriple
    source::String
    dest::String
    dep_path::String
end

function extract_sentences(file_path::String)
    triple_dicts = JSON.parsefile(file_path)["sentences"]
    map(triple_to_sentence, [ORLMTriple(d["source"], d["dest"], d["depPath"])
                             for d in triple_dicts])
end

const dep_path_tokens = r"(\||\-\>|\<\-|\w+)"
function triple_to_sentence(triple::ORLMTriple)
    cur_pos::Int = 1
    tokens::Array{String} = []
    while cur_pos <= length(triple.dep_path)
        token = match(dep_path_tokens, triple.dep_path, cur_pos).match
        push!(tokens, token)
        cur_pos += length(token)
    end
    join([triple.source, tokens[6:4:end-5]..., triple.dest], ' ')
end

if abspath(PROGRAM_FILE) == @__FILE__
    map(println, extract_sentences(ARGS[1]))
end
