module GenWorldModels

using Gen
using FunctionalCollections
using DataStructures

include("oupm_types.jl") # OUPM objects
include("oupm_moves.jl") # types for OUPM moves

include("world/world.jl")
include("mgf.jl")
include("mgfcall_map.jl")
include("lookup_or_generate.jl")
include("address_trees.jl")
include("using_world.jl")

export lookup_or_generate, UsingWorld, World, mgfcall_map

include("involution_dsl.jl")

include("object_set/object_set.jl")

# # don't export the macros; users can import them if needed
# include("macros.jl")

end