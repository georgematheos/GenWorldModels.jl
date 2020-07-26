module GenWorldModelsTests
using Gen
using GenWorldModels
using Test
using UUIDs
# include("address_trees.jl")
# # include("macros.jl")
# include("simple.jl")
# include("misc_gfi.jl")
# include("scene_decoration.jl")
# include("factorial.jl")
# # include("multiple_mgf.jl")
# include("simple_dependency_updates.jl")
# include("world_args.jl")
# include("randomized_dependency_updates.jl")

# include("oupm_types.jl")
include("oupm_involution_dsl.jl")
new_tr = run_mh_100(tr, birth_death_mh_kern, obs)
println("new: ", get_score(new_tr))
println("old: ", get_score(tr))

end # module