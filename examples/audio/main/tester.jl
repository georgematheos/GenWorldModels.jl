module Tester
include("main.jl")
using .AudioInference
trr = AudioInference.tones_with_noise(10.); nothing
initial_tr, _ = AudioInference.generate_initial_tr(trr); nothing;
function get_acc_tracker(initial_tr)
    prev_tr = initial_tr
    num_acc = 0
    function track(tr)
        if tr[:kernel => :n_tones] != prev_tr[:kernel => :n_tones]
            prev_tr = tr
            num_acc += 1
        end
    end
    get_acc() = num_acc
    (track, get_acc)
end

N_ITERS = 10
(track, get_acc) = get_acc_tracker(initial_tr)
inferred_tr = AudioInference.do_smart_bd_inference(initial_tr, N_ITERS, track)

println("Fraction accepted: ", get_acc()/N_ITERS)

end # module