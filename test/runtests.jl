module GenWorldModelsTests
using Gen
using GenWorldModels
using Test
using UUIDs
include("address_trees.jl")
# include("macros.jl")
include("simple.jl")
include("misc_gfi.jl")
include("scene_decoration.jl")
include("factorial.jl")
# include("multiple_mgf.jl")
include("simple_dependency_updates.jl")
include("world_args.jl")
include("randomized_dependency_updates.jl")

include("oupm_types.jl")

end # module