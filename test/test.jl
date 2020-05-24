module WorldModelsTests
using Gen
using Test
include("../src/WorldModels.jl")
using .WorldModels

include("simple.jl")
include("scene_decoration.jl")
include("factorial.jl")
include("multiple_mgf.jl")
include("simple_dependency_updates.jl")
include("randomized_dependency_updates.jl")

#= TODOs:
- hide metadata addr on the choicemap exposed to users
=#

end # module