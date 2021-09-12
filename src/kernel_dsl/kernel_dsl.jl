include("inference_commands.jl")
include("auto_num_change.jl")

get_world(tr::UsingWorldTrace) = tr.world
get_world(tr::GenTraceKernelDSL.TraceToken) = get_world(tr.trace)
get_world_args(tr::GenTraceKernelDSL.TraceToken) = get_world_args(tr.trace)
WorldUpdate!(tr::GenTraceKernelDSL.TraceToken, args...) = WorldUpdate!(tr.trace, args...)