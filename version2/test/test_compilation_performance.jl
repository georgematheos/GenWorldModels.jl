"""
test_compilation_performance.jl
This file provides some code to time certain types of lookups
which are supposed to be compiled so they can happen pretty close to instantly.
(For instance, getting the generative function for a given address
for a world should dispatch to a custom function which directly returns
the right generative function, for each address, so should vastly
outperform looking up the address in a dictionary.)
"""

module TestWorld
using Gen
using FunctionalCollections
using DataStructures

include("../src/world.jl")

# test performance for get_gen_fn
@gen function gen_fn1()
    return @trace(normal(0, 1), :a)
end
@gen (static) function gen_fn2()
    b = @trace(normal(0, 1), :b)
    return b
end
@gen function gen_fn3()
    return @trace(normal(0, 1), :c)
end
w = World((:a, :b, :c), (gen_fn1, gen_fn2, gen_fn3))
d = Dict(:a => gen_fn1, :b => gen_fn2, :c => gen_fn3)

function run_dict_lookups(d::Dict)
    x = 1
    for i=1:10^6
        x = d[:a]
        d[:b]
        x = d[:c]
    end
    return x
end

function run_world_lookups(w::World)
    x = 1
    for i=1:10^6
        x = get_gen_fn(w, :a)
        get_gen_fn(w, :b)
        x = get_gen_fn(w, :c)
    end
    return x
end

function run_world_call_lookups(w::World)
    x = 1
    call1 = Call{:a}(1)
    call2 = Call{:b}(2)
    call3 = Call{:c}(3)
    for i=1:10^6
        x = get_gen_fn(w, call1)
        get_gen_fn(w, call2)
        x = get_gen_fn(w, call3)
    end
    return x
end

run_dict_lookups(d)
run_world_lookups(w)
run_world_call_lookups(w)

println()
println("Looking up generative function for an address:")
println("Lookup in dictionary:")
@time run_dict_lookups(d)
println("Lookup in world (should be more than 1000x faster than the dict lookups):")
@time run_world_lookups(w)
println("Lookup in world via a `Call` object (should perform similarly to the above)")
@time run_world_call_lookups(w)

end