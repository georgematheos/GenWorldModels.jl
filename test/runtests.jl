using Gen
using GenWorldModels
using Test

include("address_trees.jl")

# core functionality tests
include("core/simple.jl")
include("core/simple_dependency_updates.jl")
include("core/world_args.jl")
include("core/multiple_mgf.jl")
include("core/randomized_dependency_updates.jl")

# a few simple examples for further behavior testing
include("examples/factorial.jl")
include("examples/scene_decoration.jl")

# OUPM functionality testing
# a few types used in several test files:
@type Aircraft
@type Timestep
@type Blip

include("oupms/oupm_types.jl")
include("oupms/simple_oupmtype_usage.jl")
include("oupms/simple_oupm_moves.jl")
include("oupms/origin_oupm_moves.jl")

include("oupms/object_sets/sibling_set_specs.jl")
include("oupms/object_sets/origin_iterated_set.jl")

include("lookup_or_generate_mapping.jl")

include("oupms/gentracekerneldsl.jl")

include("dsls/modeling_dsl_unit_tests.jl")
include("dsls/modeling_dsl_integration-seismic.jl")
include("dsls/kernel_dsl_unit_tests.jl")

include("dsls/integration-gmm.jl")