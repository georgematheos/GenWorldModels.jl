function save_num_facts(num_rels, datetime, description, num_true_facts, num_facts_over_time)
    path = joinpath(@__DIR__, "../out/saves", "run_$(datetime)_$(description)_$(num_rels)")
    io = open(path, "w")
    println(io, description)
    println(io, num_rels)
    println(io, num_true_facts)
    println(io, num_facts_over_time)
    close(io)
end

function evaluate_inference()
    datetime = Dates.format(now(), "yyyymmdd-HH_MM_SS")
    num_entities=8
    num_sentences=100
    for num_rels=2:10
        num_verbs=Int(floor(num_rels*1.5))
        global NUM_REL_PRIOR_MEAN = num_rels
        ground_truth_tr = nothing
        while ground_truth_tr === nothing
            try
                ground_truth_tr, _ = generate(generate_sentences, (num_entities, num_verbs, num_sentences), choicemap((:world => :num_relations => () => :num, num_rels)))
            catch e
                println("caught error $e while trying to generate trace for $num_rels; will try again...")
            end
        end
        true_num_facts = length(ground_truth_tr[:kernel => :facts => :facts])

        for splitmergetype in (:sdds, :smart, :dumb)
            println("Running for ", String(splitmergetype), " with $num_rels relations...")
            try
                tr = get_initial_trace(get_retval(ground_truth_tr), num_entities, num_verbs)
                num_facts_over_time = infer_num_facts(tr, splitmergetype, 1000; log_freq=100, examine_freq=10, save_freq=100)
                save_num_facts(num_rels, datetime, String(splitmergetype), true_num_facts, num_facts_over_time)
            catch e
                println("Caught error while running inference for $splitmergetype $num_rels; going to skip this iteration.")
                println("The error was $e")
            end
        end
    end
end