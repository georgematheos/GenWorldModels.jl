using PyPlot
using Statistics: mean, median, quantile, std

#########
# error #
#########

include("cond_gt_given_inf.jl")
#include("cond_ent_inf_given_gt.jl")

close("all")
figure(figsize=(5,3))
plot(sort(times_gwm), ents_gwm[sortperm(times_gwm)], label="Ours (Split/Merge)", color="blue")
plot(sort(times_van_sm), ents_van_sm[sortperm(times_van_sm)], label="Vanilla Gen (Split/Merge)", color="red")
plot(sort(times_van_generic), ents_van_generic[sortperm(times_van_generic)], label="Vanilla Gen (Generic)", color="red", linestyle="--")
gca().set_xlim(-50, 1500)
gca().set_ylim(0.75, 2.75)
xlabel("Median running time (sec.)")
ylabel("Error")
legend()
tight_layout()
savefig("entity_error_vs_time.pdf")

##############################
# running time for iteration #
##############################

num_sentences = [600, 1200, 2400, 3600, 4800]

gwm_times = [
0.36 0.449 1.529 1.532 2.464;
0.372 0.92 0.683 0.984 1.165;
0.286 0.567 1.023 1.303 2.678;
0.36 0.522 0.72 1.649 1.604;
0.443 0.453 1.065 1.483 1.583;
0.324 0.523 1.116 1.038 5.997;
0.359 0.728 0.973 1.79 1.782;
0.358 0.588 1.072 1.429 1.565
]

vgen_times = [
0.763	1.53	3.846	6.087	6.638;
0.945	1.475	2.545	6.237	7.449;
0.685	0.821	4.173	3.603	6.382;
0.999	0.813	4.289	6.075	8.663;
0.608	1.336	4.264	4.51	6.63;
0.579	1.829	2.678	5.486	5.739
]

close("all")
figure(figsize=(4,2))
plot(num_sentences, median(gwm_times, dims=[1])[:], label="Ours", color="blue")
plot(num_sentences, median(vgen_times, dims=[1])[:], label="Vanilla Gen", color="red")
xlabel("Number of sentences")
ylabel("Median time per MCMC iteration (sec.)")
legend()
tight_layout()
savefig("iteration_time_vs_num_sentences.pdf")
