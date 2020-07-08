module GenWorldModels

using Gen
using FunctionalCollections
using DataStructures

include("index_diff.jl")
include("address_filter_choicemap.jl")
include("world/world.jl")
include("lookup_or_generate.jl")
include("using_world.jl")

export lookup_or_generate, UsingWorld

# don't export the macros; users can import them if needed
# include("macros.jl")

end