module AudioInference

using Gen
using WAV
include("../tools/plotting.jl")
include("../model/gammatonegram.jl");
include("../model/time_helpers.jl");
include("../model/extra_distributions.jl");

source_params, steps, gtg_params, obs_noise = include("../params/base.jl")
sr = 2000.0
gtg_params["dB_threshold"] = 0.0
wts, = gtg_weights(sr, gtg_params);

scene_length, steps, sr = (2.0, steps, sr)
args = (scene_length, steps, sr, wts, gtg_params)

using GenWorldModels

include("worldmodel/model.jl")
include("worldmodel/inference.jl")

function vis_and_write_wave(tr, title)
    duration, _, sr, = get_args(tr)
    gram, scene_wave, = get_retval(tr)
    wavwrite(scene_wave/maximum(abs.(scene_wave)), title, Fs=sr)
    plot_gtg(gram, duration, sr, 0, 100)
end

function tones_with_noise(amp)
    cm = choicemap((:kernel => :n_tones) => 3,
        (:world => :waves => AudioSource(1) => :is_noise) => false,
        (:world => :waves => AudioSource(1) => :erb) => 10.0,
        (:world => :waves => AudioSource(1) => :onset) => 0.5,
        (:world => :waves => AudioSource(1) => :duration) => 0.3,
        (:world => :waves => AudioSource(2) => :is_noise) => false,
        (:world => :waves => AudioSource(2) => :erb) => 10.0,
        (:world => :waves => AudioSource(2) => :onset) => 1.1,
        (:world => :waves => AudioSource(2) => :duration) => 0.3,
        (:world => :waves => AudioSource(3) => :is_noise) => true,
        (:world => :waves => AudioSource(3) => :amp) => amp,
        (:world => :waves => AudioSource(3) => :onset) => 0.8,
        (:world => :waves => AudioSource(3) => :duration) => 0.3)
    tr, = generate(generate_scene, args, cm)
    return tr
end

function get_worldmodel_likelihood_tracker_and_recorder()
    likelihoods = Float64[]
    function record_worldmodel_iter!(tr)
        push!(likelihoods,
            project(tr, select(:kernel => :scene))
        )
    end
    return (likelihoods, record_worldmodel_iter!)
end

generate_initial_tr(tr) = generate(generate_scene, args, choicemap((:kernel => :scene, tr[:kernel => :scene])))

export tones_with_noise, vis_and_write_wave
export get_worldmodel_likelihood_tracker_and_recorder, generate_initial_tr
export do_generic_inference, do_birth_death_inference, do_split_merge_inference

end