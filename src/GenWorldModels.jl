module GenWorldModels

using Gen
using GenTraceKernelDSL
using FunctionalCollections
using DataStructures

include("util/setdict.jl")
include("util/unique_value_set.jl")

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

include("object_set/object_set.jl")
include("dict_map.jl")

include("dsl/modeling_dsl.jl")
export @oupm
export @kernel, MHProposal # from GenTraceKernelDSL

include("modeling_library/uniform_choice.jl")
export uniform_choice

@load_generated_functions()
end