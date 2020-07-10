module GenWorldModelsTests
using Gen
using GenWorldModels
using Test
include("address_filter_choicemap.jl")
include("simple.jl")
include("misc_gfi.jl")
include("scene_decoration.jl")
include("factorial.jl")
# include("multiple_mgf.jl")  ### WILL INCLUDE once macro syntax works
include("simple_dependency_updates.jl")
include("world_args.jl")
include("randomized_dependency_updates.jl")

end # module