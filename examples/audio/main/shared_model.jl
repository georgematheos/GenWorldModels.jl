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
    try
        noise_wave = generate_noise(transpose(reshape(fill(amp, length(times)), (length(f), length(t)))), duration, steps, sr, 1e-6)
        return embed_in_scene(scene_length, sr, noise_wave, onset)
    catch e
        println("error found while trying to generate noise for")
        display(transpose(reshape(fill(amp, length(times)), (length(f), length(t)))))
        throw(e)
    end
end

@gen function generate_single_tone(scene_length, step_size, sr)
    step_size = step_size["t"]
    erb ~ uniform(0.4, 24.0)
    onset ~ uniform(0.0, scene_length)
    duration ~ uniform(0.1, 1.0)
    if duration < 0.1
        println("IMPOSSIBLE! duration < .1")
    end
    times = get_element_gp_times([onset, onset + duration], step_size)
    wave = generate_tone(fill(erb, length(times)), fill(50.0, length(times)), duration, step_size, sr, 1.0e-6)
    return embed_in_scene(scene_length, sr, wave, onset)
end