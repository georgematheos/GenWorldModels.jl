#=
inference_commands.jl

This file contains macros for use in involutive MCMC for world models.
=#

macro objects(trace, spec)

end

macro origin(trace, object)

end
macro origin(object)

end

macro index(trace, object)

end
macro index(object)

end

macro get(expr)

end

# TODO: Should I define `set` and OUPMMoves in this file or in another file?