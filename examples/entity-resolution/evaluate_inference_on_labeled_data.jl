Tuple(s::SentenceNumeric) = (s.ent1, s.verb, s.ent2)
function run_evaluation_on_labeled_dataset(data::LabeledDataSet, datetime)
    try
        sentences = map(Tuple, data.sentences_numeric)
        tr = get_initial_trace(sentences, length(data.entity_strings), length(data.verb_strings))
        println("generated initial trace")
        num_facts_over_time = infer_num_facts(tr, :sdds, 1000; log_freq=10, examine_freq=10, save_freq=100)
        println("inferred!")
        save_num_facts(num_rels, datetime, "sdds", length(data.facts_numeric), num_facts_over_time) 
        println("saved!")
    catch e
        println("Caught error while running inference for $splitmergetype $num_rels; going to skip this iteration.")
        println("The error was $e")
    end
end
function evaluate_on_labeled_dataset(data::LabeledDataSet)
    println("evalutaing on labeled dataset...")
    datetime = Dates.format(now(), "yyyymmdd-HH_MM_SS")
    run_evaluation_on_labeled_dataset(data, datetime)
end