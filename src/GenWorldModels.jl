module GenWorldModels

using Gen
using FunctionalCollections
using DataStructures
using UUIDs

include("index_diff.jl")
include("oupm_types.jl")
include("world/world.jl")
include("mgf.jl")
include("lookup_or_generate.jl")
include("address_trees.jl")
include("using_world.jl")

export lookup_or_generate, UsingWorld, @type, World

# don't export the macros; users can import them if needed
include("macros.jl")

end