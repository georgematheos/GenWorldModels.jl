module SeismicMonitoring
using Gen
using GenWorldModels

include("constants.jl")
include("distributions.jl")
include("model.jl")

tr = simulate(generate_observations, ())
display(get_choies(tr))

end