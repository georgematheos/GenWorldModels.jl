function make_constraints(sentences::Vector{Tuple{Int, Int, Int}}, num_entities)
    constraints = choicemap(
        (:world => :num_relations => () => :num, NUM_REL_PRIOR_MEAN),
    )
    for e1=1:num_entities, e2=1:num_entities, r=1:NUM_REL_PRIOR_MEAN
        constraints[:world => :num_facts => (Relation(r), Entity(e1), Entity(e2)) => :is_true] = true
    end
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

function inference_iter(tr, num_entities, num_sentences)
    for _=1:10
        tr = update_facts_and_sparsities(tr, num_entities)

        tr = update_assignments(tr, num_sentences)
    end
    
    tr = splitmerge_updates(tr)
        
    tr
end

####
# fact & sparsity updates
####
function update_facts_and_sparsities(tr, num_entities)
    for r=1:tr[:world => :num_relations => ()]
        for e1=1:num_entities
            for e2=1:num_entities
                tr, _ = mh(tr, select(:world => :num_facts => (Relation(r), Entity(e1), Entity(e2))))
            end
        end
        tr, _ = mh(tr, select(:world => :sparsity => Relation(r)))
        for e1=1:num_entities
            for e2=1:num_entities
                tr, _ = mh(tr, select(:world => :num_facts => (Relation(r), Entity(e1), Entity(e2))))
            end
        end
    end
    tr
end

###
# assignment updates
###
struct UniformFromList <: Gen.Distribution{Any} end
uniform_from_list = UniformFromList()
function Gen.random(::UniformFromList, list)
    idx = uniform_discrete(1, length(list))
    return list[idx]
end
function Gen.logpdf(::UniformFromList, obj, list)
    if obj in list
        -log(length(list))
    else
        -Inf
    end
end

@gen function assignment_update_proposal(tr, idx)
    sentence = get_retval(tr)[idx]
    e1 = Entity(sentence[1])
    e2 = Entity(sentence[3])
    facts_for_entpair = Set([
        Fact((Relation(r), e1, e2), 1) for r=1:tr[:world => :num_relations => ()] if tr[:world => :num_facts => (Relation(r), e1, e2)]
    ])
    sampled_fact = {:kernel => :facts => :sampled_facts => idx} ~ uniform_choice(tr.world, facts_for_entpair)

    old_fact = tr[:kernel => :facts => :sampled_facts => idx]
    
    new_rel_conc = GenWorldModels.convert_to_concrete(tr.world, sampled_fact).origin[1]
    old_rel_conc = GenWorldModels.convert_to_concrete(tr.world, old_fact).origin[1]
    new_rel_abst = GenWorldModels.convert_to_abstract(tr.world, new_rel_conc)
    old_rel_abst = GenWorldModels.convert_to_abstract(tr.world, old_rel_conc)

    # resample dirichlets
    num_verbs = get_args(tr)[2]
    updated_counts = note_entity_changed_at_index(tr[:kernel => :counts], idx, old_rel_abst, new_rel_abst, sentence[2], num_verbs)

    if old_rel_conc != new_rel_conc
        {:world => :verb_prior => new_rel_conc => :prior} ~ dirichlet(updated_counts.num_mentions_per_entity[new_rel_abst] .+ DIRICHLET_PRIOR_VAL)
        if haskey(updated_counts.num_mentions_per_entity, old_rel_abst)
            {:world => :verb_prior => old_rel_conc => :prior} ~ dirichlet(updated_counts.num_mentions_per_entity[old_rel_abst] .+ DIRICHLET_PRIOR_VAL)
        end
    end
end

function metropolis_hastingsz(
    trace, proposal::GenerativeFunction, proposal_args::Tuple;
    check=false, observations=EmptyChoiceMap())
    model_args = get_args(trace)
    argdiffs = map((_) -> NoChange(), model_args)
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_choices, fwd_weight, _) = propose(proposal, proposal_args_forward)
    (new_trace, weight, _, discard) = update(trace,
        model_args, argdiffs, fwd_choices)
   # display(discard)
    proposal_args_backward = (new_trace, proposal_args...,)
    (bwd_weight, _) = assess(proposal, proposal_args_backward, discard)
    alpha = weight - fwd_weight + bwd_weight
    check && check_observations(get_choices(new_trace), observations)
    if log(rand()) < alpha
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end

function update_assignments(tr, num_sentences)
    for idx=1:num_sentences
        tr, _ = metropolis_hastingsz(tr, assignment_update_proposal, (idx,))
    end
    tr
end

####
# splitmerge updates
####

include("splitmerge.jl")

function splitmerge_updates(tr)
    tr, _ = mh(tr, dumb_splitmerge_kernel; check=true)
    return tr
end