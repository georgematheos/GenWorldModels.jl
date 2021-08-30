module GenWorldModels

# TODO: reevaluate -- what do we want to export vs what should be considered "internals"
# which should only be accessed via the DSLs?

using Gen
using GenTraceKernelDSL
using FunctionalCollections
using DataStructures

include("util/setdict.jl")
include("util/unique_value_set.jl")

### Core functionality ###
include("internals/oupm_types.jl") # OUPM objects
export @type, OUPMObject, AbstractOUPMObject, ConcreteIndexOUPMObject, ConcreteIndexAbstractOriginOUPMObject

include("internals/oupm_moves.jl") # types for OUPM moves
export Create, Delete, Split, Merge, Move, WorldUpdate

include("internals/world/world.jl")
include("internals/mgf.jl")
include("internals/mgfcall_map.jl")
include("internals/lookup_or_generate.jl")
include("internals/address_trees.jl")
include("internals/using_world.jl")

export UsingWorld, World

export lookup_or_generate
export map_lookup_or_generate, setmap_lookup_or_generate
export nocollision_setmap_lookup_or_generate, dictmap_lookup_or_generate

export to_abstract_repr, to_abstract_repr!, to_concrete_repr
export convert_to_abstract, concert_to_abstract!, concert_to_concrete
export values_to_abstract, values_to_abstract!, values_to_concrete

include("internals/object_set/object_set.jl")
export get_sibling_set, get_sibling_set_from_num
export get_origin_iterated_set
export constlen_vec

include("internals/dict_map.jl")

### Modeling DSL ###
include("modeling_dsl/modeling_dsl.jl")
export @oupm
export @kernel, MHProposal # from GenTraceKernelDSL

### Kernel DSL ###
include("kernel_dsl/kernel_dsl.jl")
export @get_number, @get, @set_number, @set, @obsmodel, @index, @abstract, @concrete, @origin, @objects, @arg, @addr
export WorldUpdate!

### Modeling Library ###
include("modeling_library/modeling_library.jl")
export uniform_choice, uniform_from_list, unnormalized_categorical

@load_generated_functions()
end