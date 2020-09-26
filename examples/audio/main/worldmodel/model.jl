@type AudioSource

include("../shared_model.jl")

@gen function generate_single_sound(world, source)
    scene_length ~ lookup_or_generate(world[:args][:scene_length])
    steps ~ lookup_or_generate(world[:args][:steps])
    sr ~ lookup_or_generate(world[:args][:sr])

    is_noise ~ bernoulli(0.4)
    if is_noise
        wave = {*} ~ generate_single_noise(scene_length, steps, sr)
    else
        wave = {*} ~ generate_single_tone(scene_length, steps, sr)
    end
    return wave
end

@gen (static) function _generate_scene(world, wts, gtg_params)
    n_tones ~ uniform_discrete(1, 4)
    
    scene_duration ~ lookup_or_generate(world[:args][:scene_length])
    audio_sr ~ lookup_or_generate(world[:args][:sr])

    waves ~ Map(lookup_or_generate)([world[:waves][AudioSource(i)] for i=1:n_tones])
    n_samples = Int(floor(scene_duration * audio_sr))
    scene_wave = reduce(+, waves; init=zeros(n_samples))
    scene_gram, = gammatonegram(scene_wave, wts, audio_sr, gtg_params)
    scene ~ noisy_matrix(scene_gram, 1.0)
    return scene_gram, scene_wave, waves
end;

generate_scene = UsingWorld(
    _generate_scene,
    :waves => generate_single_sound;
    world_args=(:scene_length, :steps, :sr)
)

@load_generated_functions()