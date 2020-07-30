module GenWorldModelsTests
using Gen
using GenWorldModels
using Test

include("address_trees.jl")

# core functionality tests
include("core/simple.jl")
include("core/simple_dependency_updates.jl")
include("core/world_args.jl")
# include("core/multiple_mgf.jl")
include("core/randomized_dependency_updates.jl")

# a few simple examples for further behavior testing
include("examples/factorial.jl")
include("examples/scene_decoration.jl")

# # OUPM functionality testing
# include("oupms/oupm_types.jl")
# include("oupms/simple_oupm_moves.jl")
# include("oupms/oupm_involution_dsl.jl")

# include("macros.jl)

end # module