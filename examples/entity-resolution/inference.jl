function make_constraints(sentences::Vector{Tuple{Int, Int, Int}}, num_entities)
    constraints = choicemap(
        (:world => :num_relations => () => :num, NUM_REL_PRIOR_MEAN),
    )
    for e1=1:num_entities, e2=1:num_entities, r=1:NUM_REL_PRIOR_MEAN
        constraints[:world => :num_facts => (Relation(r), Entity(e1), Entity(e2)) => :is_true] = true
    end
    # true_for_rel = [Set() for _=1:NUM_REL_PRIOR_MEAN]
    # for r=1:NUM_REL_PRIOR_MEAN
    #     constriants[:world => :sparsity => Relation(r) => :sparsity] = 0.2
    #     certainly_true1 = uniform_discrete(1, num_entities)
    #     certainly_true2 = uniform_discrete(1, num_entities)
    #     push!(true_for_rel
    # end
    for (i, sentence) in enumerate(sentences)
        constraints[:kernel => :rels_and_sentences => i => :verb => :verb] = sentence[2]
        rel = uniform_discrete(1, NUM_REL_PRIOR_MEAN)
        constraints[:kernel => :facts => :sampled_facts => i] = Fact((Relation(rel), Entity(sentence[1]), Entity(sentence[3])), 1)
    end
    return constraints
end

function get_initial_trace(sentences::Vector{Tuple{Int, Int, Int}}, num_entities, num_verbs)
    constraints = make_constraints(sentences, num_entities)
    tr, _ = generate(generate_sentences, (num_entities, num_verbs, length(sentences)), constraints)
    return tr
end

#############
# Inference #
#############

function infer_mean_num_rels(tr, num_iters; log_freq=num_iters + 1, save_freq=num_iters+1)
    num_rel_sum = 0
    iters_run = 0
    function examine!(tr)
        num_rel_sum += tr[:world => :num_relations => ()]
        iters_run += 1
    end
    try
        run_inference!(tr, examine!, num_iters; log_freq=log_freq, examine_freq=1, save_freq=save_freq)
        return num_rel_sum/iters_run
    catch e
        println("Caught error ", e)
        println("Managed to run $iters_run iterations.  Mean num relations at this point was ", num_rel_sum/iters_run)
    end
end

function save!(tr, datetime, i)
    path = joinpath(@__DIR__, "saves", "infchoicemap_$(datetime)__$i")
    # println("dynamic copy : ")
    # println( Gen.deep_dynamic_copy(get_choices(tr)))
    serialize(path, Gen.deep_dynamic_copy(get_choices(tr)))
end

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

# `examine!(tr)` updates some sort of state; is called at the end of every iteration which is a multiple of `examine_freq`
# saves the trace at `examples/entity-resolution/saves/infchoicemap_STARTDATETIME__iter`
function run_inference!(tr, examine!, num_iters; log_freq=num_iters+1, examine_freq=1, save_freq=num_iters+1)
    datetime = Dates.format(now(), "yyyymmdd-HH_MM_SS")
    (num_entities, _, num_sentences) = get_args(tr)
    tracker = AccTracker(0, 0, 0, 0, 0, 0, 0)
    for i=1:num_iters
        if i % log_freq === 0
            println("Running iter $i...")
            display(tracker)
        end

        tr = inference_iter(tr, tracker)

        if i % examine_freq === 0
            examine!(tr)
        end
        if i % save_freq === 0
            save!(tr, datetime, i)
        end
    end
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
    # println("Fact update: ", acc ? "ACCEPTED" : "REJECTED")
    return tr
end

include("splitmerge.jl")

function splitmerge_update(tr, acc_tracker)
    new_tr, acc = mh(tr, dumb_split_smart_merge_kernel; check=false)
    
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
    {:world => :sparsity => Relation(idx) => :sparsity} ~ beta(BETA_PRIOR[1] + num_true, BETA_PRIOR[2] + num_entities^2)
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

function inference_iter(tr, acc_tracker)
    type = categorical([0.6, 0.1, 0.3])
    if type == 1
        tr = update_random_fact(tr, acc_tracker)
    elseif type == 2
        tr = sparsity_update(tr, acc_tracker)
    else
        tr = splitmerge_update(tr, acc_tracker)
    end
    return tr
end

# function inference_iter(tr, num_entities, num_sentences)


#     for _=1:10
#         tr = update_facts_and_sparsities(tr, num_entities)

#         tr = update_assignments(tr, num_sentences)
#     end
    
#     tr = splitmerge_updates(tr)
        
#     tr
# end

# ####
# # fact & sparsity updates
# ####

# function update_facts_and_sparsities(tr, num_entities)
#     for r=1:tr[:world => :num_relations => ()]
#         for e1=1:num_entities
#             for e2=1:num_entities
#                 tr, _ = mh(tr, select(:world => :num_facts => (Relation(r), Entity(e1), Entity(e2))))
#             end
#         end
#         tr, _ = mh(tr, select(:world => :sparsity => Relation(r)))
#         for e1=1:num_entities
#             for e2=1:num_entities
#                 tr, _ = mh(tr, select(:world => :num_facts => (Relation(r), Entity(e1), Entity(e2))))
#             end
#         end
#     end
#     tr
# end

# ###
# # assignment updates
# ###
# struct UniformFromList <: Gen.Distribution{Any} end
# uniform_from_list = UniformFromList()
# function Gen.random(::UniformFromList, list)
#     idx = uniform_discrete(1, length(list))
#     return list[idx]
# end
# function Gen.logpdf(::UniformFromList, obj, list)
#     if obj in list
#         -log(length(list))
#     else
#         -Inf
#     end
# end

# @gen function assignment_update_proposal(tr, idx)
#     sentence = get_retval(tr)[idx]
#     e1 = Entity(sentence[1])
#     e2 = Entity(sentence[3])
#     facts_for_entpair = Set([
#         Fact((Relation(r), e1, e2), 1) for r=1:tr[:world => :num_relations => ()] if tr[:world => :num_facts => (Relation(r), e1, e2)]
#     ])
#     sampled_fact = {:kernel => :facts => :sampled_facts => idx} ~ uniform_choice(tr.world, facts_for_entpair)

#     old_fact = tr[:kernel => :facts => :sampled_facts => idx]
    
#     new_rel_conc = GenWorldModels.convert_to_concrete(tr.world, sampled_fact).origin[1]
#     old_rel_conc = GenWorldModels.convert_to_concrete(tr.world, old_fact).origin[1]
#     new_rel_abst = GenWorldModels.convert_to_abstract(tr.world, new_rel_conc)
#     old_rel_abst = GenWorldModels.convert_to_abstract(tr.world, old_rel_conc)

#     # resample dirichlets
#     num_verbs = get_args(tr)[2]
#     updated_counts = note_entity_changed_at_index(tr[:kernel => :counts], idx, old_rel_abst, new_rel_abst, sentence[2], num_verbs)

#     if old_rel_conc != new_rel_conc
#         {:world => :verb_prior => new_rel_conc => :prior} ~ dirichlet(updated_counts.num_mentions_per_entity[new_rel_abst] .+ DIRICHLET_PRIOR_VAL)
#         if haskey(updated_counts.num_mentions_per_entity, old_rel_abst)
#             {:world => :verb_prior => old_rel_conc => :prior} ~ dirichlet(updated_counts.num_mentions_per_entity[old_rel_abst] .+ DIRICHLET_PRIOR_VAL)
#         end
#     end
# end

# function update_assignments(tr, num_sentences)
#     for idx=1:num_sentences
#         tr, _ = mh(tr, assignment_update_proposal, (idx,))
#     end
#     tr
# end

####
# splitmerge updates
####

