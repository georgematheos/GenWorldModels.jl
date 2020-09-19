mutable struct AccTracker
    num_acc_split::Int
    num_acc_merge::Int
    num_rejected_splitmerge::Int
    num_rel_assoc_acc::Int
    num_rel_assoc_rej::Int
end
function Base.show(io::IO, trk::AccTracker)
    println(io, "  ACCEPTED SPLIT       : ", trk.num_acc_split)
    println(io, "  ACCEPTED MERGE       : ", trk.num_acc_merge)
    println(io, "  REJECTED Splitmerge  : ", trk.num_rejected_splitmerge)
    println(io, "  NUM ACC Rel ASSOC    : ", trk.num_rel_assoc_acc)
    println(io, "  NUM REJ Rel ASSOC    : ", trk.num_rel_assoc_rej)
end

function get_entpair_to_indices(entpairs)
    etoi = Dict()
    for (i, pair) in enumerate(entpairs)
        if haskey(etoi, pair)
            push!(etoi[pair], i)
        else
            etoi[pair] = [i]
        end
    end
    etoi
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
    tracker = AccTracker(0, 0, 0, 0, 0)
    entpair_to_indices = get_entpair_to_indices(entpairs(initial_tr))

    inference_iter(tr) = splitmerge_inference_iter(tr, tracker, splitmerge_type, entpair_to_indices)
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

"""
    splitmerge_all_facts_referenced_inference_iter(tr, acc_tracker, splitmerge_type)

An iteration for a Markov Chain over the state space, given the observations
and that every true fact appears in at least one sentence.  Will randomly be either
a split/merge update or an update to which relation is assigned to each sentence.

`acc_tracker` is an AcceptanceTracker to accumulate acceptance counts into.
`splitmerge_type` should be `:smart`, `:dumb`, or `:sdds`.
`entpair_to_indices` should be a dictionary from entpair `(e1, e2)` to a collection
of all the indices in which these entpairs appear.
"""
function splitmerge_inference_iter(tr, acc_tracker, splitmerge_type, entpair_to_indices)
    do_splitmerge = bernoulli(0.2)
    if do_splitmerge
       tr = splitmerge_update(tr, acc_tracker, splitmerge_type)
    else
        tr = update_random_sentence_relation(tr, acc_tracker, entpair_to_indices)
    end
       
    return tr
end

include("splitmerge.jl")
function splitmerge_update(tr, acc_tracker, splitmerge_type)
    @assert splitmerge_type === :sdds "Other splitmerge types not implemented"
    new_tr, acc = mh(tr, sdds_splitmerge_kernel; check=false)
    
    diff = new_tr[:world => :num_relations => ()] - tr[:world => :num_relations => ()]
    if diff > 0
        acc_tracker.num_acc_split += 1
    elseif diff < 0
        acc_tracker.num_acc_merge += 1
    else
        acc_tracker.num_rejected_splitmerge += 1
    end

    return new_tr
end

@dist list_categorical(probs, list) = list[categorical(probs)]
macro exactly(x)
    quote uniform_from_list([$x]) end
end
@gen function relation_update_proposal(tr, sentence_idx, entpair_to_indices)
    nrels = tr[:world => :num_relations => ()]
    entpair = entpairs(tr)[sentence_idx]
    indices = entpair_to_indices[entpair]
    true_rels_abstract = Set(tr[:kernel => :sampled_facts][ind].rel for ind in indices if ind !== sentence_idx)
    true_rel_indices = Set(GenWorldModels.convert_to_concrete(tr.world, rel).idx for rel in true_rels_abstract)
    unnormalized_probs = [rel_idx in true_rel_indices ? 2. : 1. for rel_idx=1:nrels]
    probs = normalize(unnormalized_probs)
    new_rel_idx ~ categorical(probs)
    old_rel_idx = GenWorldModels.convert_to_concrete(tr.world, tr[:kernel => :sampled_facts][sentence_idx].rel).idx
    return (new_rel_idx, old_rel_idx, true_rel_indices, entpair)
end
@involution function relation_update_inv(_, proposal_args, proposal_retval)
    (new_rel_idx, old_rel_idx, true_rel_indices, entpair) = proposal_retval
    (sentence_idx, _) = proposal_args
    new_fact = Fact(Relation(new_rel_idx), entpair...)
    @write_discrete_to_model(:kernel => :sampled_facts => :sampled_facts => sentence_idx, new_fact)

    if !(new_rel_idx in true_rel_indices)
        @write_discrete_to_model(:kernel => :sampled_facts => :all_facts => :rels_to_facts => Relation(new_rel_idx) => :true_entpairs => entpair, true)
    end

    if !(old_rel_idx in true_rel_indices) && old_rel_idx != new_rel_idx
        @write_discrete_to_model(:kernel => :sampled_facts => :all_facts => :rels_to_facts => Relation(old_rel_idx) => :true_entpairs => entpair, false)
    end

    @write_discrete_to_proposal(:new_rel_idx, old_rel_idx)
end

function update_random_sentence_relation(tr, acc_tracker, entpair_to_indices)
    idx = uniform_discrete(1, length(get_retval(tr)[1]))
    tr, acc = mh(tr, relation_update_proposal, (idx, entpair_to_indices), relation_update_inv, check=false)
    if acc
        acc_tracker.num_rel_assoc_acc += 1
    else
        acc_tracker.num_rel_assoc_rej += 1
    end
    return tr
end