function generic_no_num_change_inference_iter(tr)
    for j = 1:tr[:kernel => :n_tones]
        tr, _ = mh(tr, select(:world => :waves => AudioSource(j)))
        if tr[:world => :waves => AudioSource(j) => :is_noise]
            tr, _ = mh(tr, select(:world => :waves => AudioSource(j) => :amp))
        else
            tr, _ = mh(tr, select(:world => :waves => AudioSource(j) => :erb))
        end
        if bernoulli(0.5)
            tr, _ = mh(tr, select(
                :world => :waves => AudioSource(j) => :onset,
                :world => :waves => AudioSource(j) => :duration
            ))
        else
            tr, _ = mh(tr, select(:world => :waves => AudioSource(j) => :onset))
            tr, _ = mh(tr, select(:world => :waves => AudioSource(j) => :duration))
        end
    end
    return tr
end

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

function run_inference(initial_tr, run_iter, num_iters)
    tr = initial_tr
    for _=1:num_iters
        tr = run_iter(tr)
    end
    return tr
end

function get_initial_tr(trr)
    observations = choicemap((:kernel => :scene, trr[:kernel => :scene]))
    initial_tr, = generate(generate_scene, args, observations)
    return initial_tr
end