module GenWorldModels

using Gen
using FunctionalCollections
using DataStructures

include("oupm_types.jl") # exports OUPM-related items

include("index_diff.jl")
include("world/world.jl")
include("mgf.jl")
include("lookup_or_generate.jl")
include("address_trees.jl")
include("using_world.jl")

export lookup_or_generate, UsingWorld, World

# include("involution_dsl.jl")

# # don't export the macros; users can import them if needed
# include("macros.jl")

end