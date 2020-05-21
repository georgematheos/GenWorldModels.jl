module WorldModelsTests
using Gen
using Test
include("../src/WorldModels.jl")
using .WorldModels

include("simple.jl")
include("scene_decoration.jl")
include("factorial.jl")

#= TODOs:
- test recursion
- test updates changing dependency structure
- test correct error-throwing behavior
- hide metadata addr on the choicemap exposed to users
=#

end # module