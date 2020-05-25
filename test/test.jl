module WorldModelsTests
using Gen
using Test
include("../src/WorldModels.jl")
using .WorldModels

include("address_filter_choicemap.jl")
include("simple.jl")
include("misc_gfi.jl")
include("scene_decoration.jl")
include("factorial.jl")
include("multiple_mgf.jl")
include("simple_dependency_updates.jl")
include("randomized_dependency_updates.jl")

end # module