@gen function duration_drift(tr, i)
    prev_duration = tr[:world => :waves => AudioSource(i) => :duration]
    minnew = max(0.1, prev_duration - 0.05)
    maxnew = min(1, prev_duration + 0.05)
    {:world => :waves => AudioSource(i) => :duration} ~ uniform(minnew, maxnew)
    # if prev_duration < 0.7
    #     {:world => :waves => AudioSource(i) => :duration} ~ shifted_folded_normal(0.1, prev_duration, 0.04)
    # else
    #     {:world => :waves => AudioSource(i) => :duration} ~ uniform(0.6, 1.)
    # end
end
@gen function onset_drift(tr, i)
    prev_onset = tr[:world => :waves => AudioSource(i) => :onset]
    scene_length = get_args(tr)[1]
    minnew = max(0.1, prev_onset - 0.05)
    maxnew = min(scene_length, prev_onset + 0.05)
    {:world => :waves => AudioSource(i) => :onset} ~ uniform(minnew, maxnew)
    # if prev_onset < scene_length - 0.3
    #     {:world => :waves => AudioSource(i) => :onset} ~ shifted_folded_normal(0.1, prev_onset, 0.04)
    # else
    #     {:world => :waves => AudioSource(i) => :onset} ~ uniform(scene_length - 0.1, scene_length)
    # end
end
@gen function param_drift(tr, i)
    if tr[:world => :waves => AudioSource(i) => :is_noise]
        prev_amp = tr[:world => :waves => AudioSource(i) => :amp]
        {:world => :waves => AudioSource(i) => :amp} ~ normal(prev_amp, 0.5)
    else
        prev_erb = tr[:world => :waves => AudioSource(i) => :erb]
        minnew = max(0.4, prev_erb - 0.8)
        maxnew = min(37, prev_erb + 0.8)
        {:world => :waves => AudioSource(i) => :erb} ~ uniform(minnew, maxnew)
    end
end

# TODO: proposal to move start without changing endpoint
# @gen function start_drift_fixedend_prop(tr, i)
#     prev_duration = tr[:world => :waves => AudioSource(i) => :duration]
#     new_duration ~ shifted_folded_normal(0.1, prev_dur, 0.04)
# end
# @oupm_involution start_drift_fixedend_inv (old, fwd) to (new, bwd) begin
#     @copy(fwd[:new_duration], new[:world => :])
# end

function drift_inference_iter(tr)
    for j = 1:tr[:kernel => :n_tones]
        for proposal in (param_drift, duration_drift, onset_drift)
            tr, _ = mh(tr, proposal, (j,))
        end
    end
    return tr
end
