module SeismicMonitoring
using Gen
using GenWorldModels

include("constants.jl")
include("distributions.jl")
include("model.jl")
include("nick_model_constraints.jl")

tr = simulate(generate_observations, ())
open("choicemap.txt", "w") do io
    show(io, MIME("text/plain"), get_choices(tr))
end

end