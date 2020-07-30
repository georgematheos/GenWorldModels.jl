
using WAV
include("../tools/plotting.jl");
include("../model/model.jl");
include("../model/gammatonegram.jl");
include("../model/time_helpers.jl");

include("../../gen-memoization/src/WorldModels.jl")
using .WorldModels

source_params, steps, gtg_params, obs_noise = include("../params/base.jl")
sr = 2000.0
gtg_params["dB_threshold"] = 0.0
wts, = gtg_weights(sr, gtg_params);

function embed_in_scene(scene_length, sr, wave, onset)
  n_samples = Int(floor(sr * scene_length))
  scene_wave = zeros(n_samples)
  sample_start = max(1, Int(floor(onset * sr)))
  sample_finish = min(sample_start + length(wave), length(scene_wave))
  scene_wave[sample_start:sample_finish-1] = wave[1:length(sample_start:sample_finish-1)]
  return scene_wave
end

@gen function generate_single_noise(scene_length, steps, sr)
  onset ~ uniform(0, scene_length)
  duration ~ uniform(0.1, 1.0)
  amp ~ normal(10.0, 8.0)
  times, t, f = get_gp_spectrotemporal([onset, onset+duration], steps, sr)
  noise_wave = generate_noise(transpose(reshape(fill(amp, length(times)), (length(f), length(t)))), duration, steps, sr, 1e-6)
  return embed_in_scene(scene_length, sr, noise_wave, onset)
end;

@gen function generate_single_tone(scene_length, step_size, sr)
  step_size = step_size["t"]
  erb ~ uniform(0.4, 37.0)
  onset ~ uniform(0.0, scene_length)
  duration ~ uniform(0.1, 1.0)
  times = get_element_gp_times([onset, onset + duration], step_size)
  wave = generate_tone(fill(erb, length(times)), fill(50.0, length(times)), duration, step_size, sr, 1.0e-6)
  return embed_in_scene(scene_length, sr, wave, onset)
end;

@gen function generate_single_sound(world, tup)
    source_idx, scene_length, steps, sr = tup
    is_noise ~ bernoulli(0.4)
    if is_noise
        wave = {*} ~ generate_single_noise(scene_length, steps, sr)
    else
        wave = {*} ~ generate_single_tone(scene_length, steps, sr)
    end
    return wave
end
#generate_sounds = Map(generate_single_sound);

@gen (static) function _generate_scene(world, scene_duration, wts, audio_sr, steps, gtg_params)
  n_tones ~ uniform_discrete(1, 4)
  waves ~ Map(lookup_or_generate)([world[:waves][(i, scene_duration, steps, audio_sr)] for i=1:n_tones])
  n_samples = Int(floor(scene_duration * audio_sr))
  scene_wave = reduce(+, waves; init=zeros(n_samples))
  scene_gram, = gammatonegram(scene_wave, wts, audio_sr, gtg_params)
  scene ~ noisy_matrix(scene_gram, 1.0)
  return scene_gram, scene_wave, waves
end;

generate_scene = UsingWorld(_generate_scene, :waves => generate_single_sound)

@load_generated_functions

function vis_and_write_wave(tr, title)
  duration, _, sr, = get_args(tr)
  gram, scene_wave, = get_retval(tr)
  wavwrite(scene_wave/maximum(abs.(scene_wave)), title, Fs=sr)
  plot_gtg(gram, duration, sr, 0, 100)
end

scene_length, steps, sr = (2.0, steps, sr)
subargs = (scene_length, steps, sr)
args = (scene_length, wts, sr, steps, gtg_params)

tr = simulate(generate_scene, args);
vis_and_write_wave(tr, "simulated_scene.wav")
