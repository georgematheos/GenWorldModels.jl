module GenWorldModels

using Gen
using FunctionalCollections
using DataStructures

include("util/setdict.jl")

include("oupm_types.jl") # OUPM objects
include("oupm_moves.jl") # types for OUPM moves

include("world/world.jl")
include("mgf.jl")
include("mgfcall_map.jl")
include("lookup_or_generate.jl")
include("address_trees.jl")
include("using_world.jl")

export UsingWorld, World

export lookup_or_generate
export map_lookup_or_generate, setmap_lookup_or_generate
export nocollision_setmap_lookup_or_generate, dictmap_lookup_or_generate

export to_abstract_repr, to_abstract_repr!, to_concrete_repr
export convert_to_abstract, concert_to_abstract!, concert_to_concrete
export values_to_abstract, values_to_abstract!, values_to_concrete

include("involution_dsl.jl")

include("object_set/object_set.jl")
include("dict_map.jl")

include("dsl/modeling_dsl.jl")
export @oupm

@load_generated_functions()

# # don't export the macros; users can import them if needed
# include("macros.jl")

end