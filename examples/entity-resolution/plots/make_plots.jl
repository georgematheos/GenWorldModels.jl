using PyPlot
using Statistics: mean, median, quantile, std

include("cond_gt_given_inf.jl")
#include("cond_ent_inf_given_gt.jl")

close("all")
figure(figsize=(4,2))
plot(sort(times_gwm), ents_gwm[sortperm(times_gwm)], label="Gen World Models (Split/Merge)", color="blue")
plot(sort(times_van_sm), ents_van_sm[sortperm(times_van_sm)], label="Vanilla Gen (Split/Merge)", color="red")
plot(sort(times_van_generic), ents_van_generic[sortperm(times_van_generic)], label="Vanilla Gen (Generic)", color="red", linestyle="--")
gca().set_xlim(-50, 1500)
gca().set_ylim(0.75, 2.75)
xlabel("Median running time (sec.)")
ylabel("Error")
tight_layout()
savefig("entity_error_vs_time.pdf")
