mutable struct AccTracker
    num_acc_split::Int
    num_acc_merge::Int
    num_rejected_splitmerge::Int
    num_acc_fact_update::Int
    num_rej_fact_update::Int
    num_acc_sparsity::Int
    num_rej_sparsity::Int
end
function Base.show(io::IO, trk::AccTracker)
    println(io, "  ACCEPTED SPLIT       : ", trk.num_acc_split)
    println(io, "  ACCEPTED MERGE       : ", trk.num_acc_merge)
    println(io, "  REJECTED Splitmerge  : ", trk.num_rejected_splitmerge)
    println(io, "  NUM ACC FACT UPDATE  : ", trk.num_acc_fact_update)
    println(io, "  NUM REJ FACT UPDATE  : ", trk.num_rej_fact_update)
    println(io, "  NUM ACC SPARSITY     : ", trk.num_acc_sparsity)
    println(io, "  NUM REJ  SPARSITY    : ", trk.num_rej_sparsity)
end

"""
    run_tracked_splitmerge_inference!(initial_tr, num_iters, examine! [; examine_freq=1, splitmerge_type=:sdds, log=false, log_freq=1])

Runs splitmerge inference, tracking acceptance ratios.  `splitmerge_type` kwarg is `:sdds` by default;
may also be `:smart` or `:dumb`.
If kwarg `log=true`, will log the inference tracker every `log_iters`.
"""
function run_tracked_splitmerge_inference!(initial_tr, num_iters, examine!;
    examine_freq=1,
    splitmerge_type=:sdds,
    log=false,
    log_freq=1
)
    tracker = AccTracker(0, 0, 0, 0, 0, 0, 0)
    inference_iter(tr) = splitmerge_inference_iter(tr, tracker, splitmerge_type)
    if log
        function our_examine!(i, tr)
            if i % examine_freq === 0
                examine!(i, tr)
            end
            if i % log_freq === 0
                println("Just ran iteration $i.")
                println("Current acceptance tracker state:")
                show(tracker)
            end
        end
        our_freq = gcd(examine_freq, log_freq)
    else
        out_examine! = examine!
        our_freq = examine_freq
    end
    run_inference!(initial_tr, inference_iter, num_iters, our_examine!; examine_freq=our_freq)
end

"""
    run_inference!(initial_tr, inference_iter, num_iters, examine!; examine_freq=1)

Runs the given `inference_iter` `num_iters` times; calls `examine!(iteration_num, current_tr)`
every `examine_freq` iterations.  Returns the trace after running the inference.
"""
function run_inference!(initial_tr, inference_iter, num_iters, examine!; examine_freq=1)
    tr = initial_tr
    for i=1:num_iters
        tr = inference_iter(tr)
        if i % examine_freq === 0
            examine!(i, tr)
        end
    end
    return tr
end

function update_random_fact(tr, acc_tracker)
    num_ents = get_args(tr)[1]
    rel = uniform_discrete(1, tr[:world => :num_relations => ()])
    ent1 = uniform_discrete(1, num_ents)
    ent2 = uniform_discrete(1, num_ents)
    tr,acc = mh(tr, select(:world => :num_facts => (Relation(rel), Entity(ent1), Entity(ent2)) => :is_true))
    if acc
        acc_tracker.num_acc_fact_update += 1
    else
        acc_tracker.num_rej_fact_update += 1
    end
    return tr
end

include("splitmerge.jl")

function splitmerge_update(tr, acc_tracker, splitmerge_type)
    if splitmerge_type == :sdds
        new_tr, acc = mh(tr, sdds_splitmerge_kernel; check=false)
    elseif splitmerge_type == :smart
        new_tr, acc = mh(tr, smart_splitmerge_kernel; check=false)
    elseif splitmerge_type == :dumb
        new_tr, acc = mh(tr, dumb_splitmerge_kernel; check=false)
    end
    
    diff = new_tr[:world => :num_relations => ()] - tr[:world => :num_relations => ()]
    if diff > 0
        acc_tracker.num_acc_split += 1
    elseif diff < 0
        acc_tracker.num_acc_merge += 1
    else
        acc_tracker.num_rejected_splitmerge += 1
    end

    new_tr
end

@gen function sparsity_proposal(tr, idx)
    num_entities = get_args(tr)[1]
    facts = tr[:kernel => :facts => :facts => :facts_per_rel => idx]
    num_true = length(facts)
    {:world => :sparsity => Relation(idx) => :sparsity} ~ beta(BETA_PRIOR[1] + num_true, BETA_PRIOR[2] + (num_entities^2 - num_true))
end

function sparsity_update(tr, acc_tracker)
    idx = uniform_discrete(1, tr[:world => :num_relations => ()])
    tr, acc = mh(tr, sparsity_proposal, (idx,))
    if acc
        acc_tracker.num_acc_sparsity += 1
    else
        acc_tracker.num_rej_sparsity += 1
    end
    tr
end

@gen function dirichlet_proposal(tr, idx)
    num_verbs = get_args(tr)[2]
    nmpe = tr[:kernel => :counts].num_mentions_per_entity
    abst = GenWorldModels.convert_to_abstract(tr.world, idx)
    count = haskey(nmpe, abst) ? nmpe[abst] : zeros(Int, num_verbs)
    {:world => :verb_prior => Relation(idx) => :prior} ~ dirichlet(count .+ DIRICHLET_PRIOR_VAL)
end
function dirichlet_update(tr, acc_tracker)
    idx = uniform_discrete(1, tr[:world => :num_relations => ()])
    if has_value(get_choices(tr), :world => :verb_priors => Relation(idx) => :prior)
        tr, acc = mh(tr, dirichlet_proposal, (idx,))
        if !acc
            println("dirichlet prop not accepted!")
        end
        return tr
    end
    return tr
end

function splitmerge_inference_iter(tr, acc_tracker, splitmerge_type)
    type = categorical([0.5, 0.1, 0.3, 0.1])
    # type = 3
    if type == 1
        tr = update_random_fact(tr, acc_tracker)
    elseif type == 2
        tr = sparsity_update(tr, acc_tracker)
    elseif type == 3
        tr = splitmerge_update(tr, acc_tracker, splitmerge_type)
    else
        tr = dirichlet_update(tr, acc_tracker)
    end
    return tr
end