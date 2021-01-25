include("smart_birth_death.jl")
include("smart_split_merge2.jl")

@dist function uniform_from_list(list)
    idx = uniform_discrete(1, length(list))
    list[idx]
end
singleton(x) = [x]
@dist exactly(x) = singleton(x)[categorical([1.0])]

@gen function duration_drift(tr, i)
    prev_duration = tr[:world => :waves => AudioSource(i) => :duration]
    scene_length = get_args(tr)[1]
    {:world => :waves => AudioSource(i) => :duration} ~ truncated_normal(
        prev_duration, DURATION_STD, MIN_DURATION(scene_length), MAX_DURATION(scene_length)
    )
end
@gen function onset_drift(tr, i)
    prev_onset = tr[:world => :waves => AudioSource(i) => :onset]
    scene_length = get_args(tr)[1]
    {:world => :waves => AudioSource(i) => :onset} ~ truncated_normal(
        prev_onset, ONSET_STD, MIN_ONSET(scene_length), MAX_ONSET(scene_length)
    )
end
@gen function param_drift(tr, i)
    if tr[:world => :waves => AudioSource(i) => :is_noise]
        prev_amp = tr[:world => :waves => AudioSource(i) => :amp]
        {:world => :waves => AudioSource(i) => :amp} ~ normal(prev_amp, AMP_STD)
    else
        prev_erb = tr[:world => :waves => AudioSource(i) => :erb]
        {:world => :waves => AudioSource(i) => :erb} ~ truncated_normal(prev_erb, ERB_STD, MIN_ERB, MAX_ERB)
    end
end
@gen function start_drift(tr, i)
    prev_onset = tr[:world => :waves => AudioSource(i) => :onset]
    prev_duration = tr[:world => :waves => AudioSource(i) => :duration]
    scene_length = get_args(tr)[1]

    new_onset = {:world => :waves => AudioSource(i) => :onset} ~ truncated_normal(
        prev_onset, ONSET_STD, MIN_ONSET(scene_length), MAX_ONSET(scene_length)
    )
    old_endtime = prev_onset + prev_duration
    new_duration = old_endtime - new_onset
    {:world => :waves => AudioSource(i) => :duration} ~ exactly(new_duration)
end

@gen function switch_source_type(tr, i)
    was_noise = tr[:world => :waves => AudioSource(i) => :is_noise]
    {:world => :waves => AudioSource(i) => :is_noise} ~ exactly(!was_noise)
    if was_noise
        {:world => :waves => AudioSource(i) => :erb} ~ uniform(0.4, 24.0)
    else
        {:world => :waves => AudioSource(i) => :amp} ~ normal(10.0, 8.0)
    end
end

function drift_pass(tr)
    for i=1:num_sources(tr)
        tr, _ = mh(tr, duration_drift, (i,))
        tr, _ = mh(tr, onset_drift, (i,))
        tr, _ = mh(tr, param_drift, (i,))
        tr, _ = mh(tr, switch_source_type, (i,))
    end
    return tr
end

function drift_smartbd_iter(tr)
    tr, _ = mh(tr, smart_birth_death_mh_kern; check=false)
    tr = drift_pass(tr)
    return tr
end

function drift_smartbd_inference(tr, iters, record_iter! = identity)
    for _=1:iters
        tr = drift_smartbd_iter(tr)
        record_iter!(tr)
    end
    return tr
end

function splitmerge(tr)
    tr, acc = metropolis_hastings(tr, smart_splitmerge_mh_kern; check=false)#, logfwdchoices=true, logscores=true)
    return tr
end

function drift_smartsmbd_iter(tr)
    tr, _ = mh(tr, smart_birth_death_mh_kern; check=false)
    tr = drift_pass(tr)
    tr = splitmerge(tr)
    return tr
end

function drift_smartsmbd_inference(tr, iters, record_iter! = identity)
    for _=1:iters
        tr = drift_smartsmbd_iter(tr)
        record_iter!(tr)
    end
    return tr
end

#####
# For debugging
#####
function (translator::GenWorldModels.OUPMMHKernel)(prev_model_trace::Trace; check=false, observations=EmptyChoiceMap(), logscores=false, logfwdchoices=false)
    # simulate from auxiliary program
    forward_proposal_trace = simulate(translator.q, (prev_model_trace, translator.q_args...,))

    # apply trace transform
    (new_model_trace, log_model_weight, backward_proposal_trace, log_abs_determinant, regenerated_vals) = GenWorldModels.symmetric_trace_translator_run_transform(
        translator.f, prev_model_trace, forward_proposal_trace, translator.q, translator.q_args)

    # compute log weight
    forward_proposal_score = get_score(forward_proposal_trace)
    backward_proposal_score = get_score(backward_proposal_trace)
    log_weight = log_model_weight + backward_proposal_score - forward_proposal_score + log_abs_determinant

    if logscores
        score_delta = get_score(new_model_trace) - get_score(prev_model_trace)
        println("  score delta / weight: $score_delta / $log_model_weight")
        println("  bwd         / fwd   : $backward_proposal_score / $forward_proposal_score")
        println("  overall weight : $log_weight")
    end
    if logfwdchoices
        display(get_choices(forward_proposal_trace))
    end

    if check
        Gen.check_observations(get_choices(new_model_trace), observations)
        forward_proposal_choices = get_choices(forward_proposal_trace)
        (prev_model_trace_rt, _, forward_proposal_trace_rt, _) = GenWorldModels.symmetric_trace_translator_run_transform(
            translator.f, new_model_trace, backward_proposal_trace, translator.q, translator.q_args; regeneration_constraints=regenerated_vals)
        GenWorldModels.check_round_trip(
            prev_model_trace, prev_model_trace_rt,
            forward_proposal_trace, forward_proposal_trace_rt)
    end

    return (new_model_trace, log_weight)
end

function Gen.metropolis_hastings(trace, apply_oupm_move::OUPMMHKernel; check=false, observations=EmptyChoiceMap(), logscores=false, logfwdchoices=false)
    (new_tr, log_weight) = apply_oupm_move(trace; check, observations, logscores, logfwdchoices)
    if log(rand()) <= log_weight
        (new_tr, true)
    else
        (trace, false)
    end
end