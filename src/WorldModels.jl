module WorldModels

using Gen
using FunctionalCollections
using DataStructures

include("index_diff.jl")
include("world/world.jl")
include("lookup_or_generate.jl")
include("using_world.jl")

export lookup_or_generate, UsingWorld

end