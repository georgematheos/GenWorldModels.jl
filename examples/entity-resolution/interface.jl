function sentences_string_to_numeric(sentences::Array{Tuple{String, String, String}}, entity_strings::Vector{String}, verbs::Vector{String})
    numeric_sentences = Tuple{Int, Int, Int}[]
    for sentence in sentences
        obj1, verb, obj2 = sentence
        obj1_indices = findall(object_strings .== obj1)
        verb_indices = findall(verbs .== verb)
        obj2_indices = findall(object_strings .== obj2)
        
        if length(obj1_indices) != 1
            error("Must be exactly one occurrence of " * obj1 * " in `object_strings` since it occurs in a sentence.")
        end
        if length(verb_indices) != 1
            error("Must be exactly one occurrence of " * verb * " in `verb` since it occurs in a sentence.")
        end
        if length(obj2_indices) != 1
            error("Must be exactly one occurrence of " * obj2 * " in `object_strings` since it occurs in a sentence.")
        end
        
        push!(numeric_sentences, (obj1_indices[1], verb_indices[1], obj2_indices[1]))
    end
    
    numeric_sentences
end