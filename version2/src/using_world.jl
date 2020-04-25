#########
# Trace #
#########

struct UsingWorldTrace{V, Tr} <: Gen.Trace
    kernel_tr::Tr
    world::World
    score::Float64
    args::Tuple
    gen_fn::GenerativeFunction{V, UsingMemoizedTrace{V, Tr}}
end

Gen.get_args(tr::UsingWorldTrace) = tr.args
Gen.get_retval(tr::UsingWorldTrace) = get_retval(tr.kernel_tr)
Gen.get_score(tr::UsingWorldTrace) = tr.score
Gen.get_gen_fn(tr::UsingWorldTrace) = tr.gen_fn
Gen.project(tr::UsingWorldTrace, sel::Selection) = error("Not implemented")
function Gen.get_choices(tr::UsingWorldTrace)
    leaf_nodes = NamedTuple()
    internal_nodes = (kernel=get_choices(tr.kernel_tr), world=get_choices(tr.world))
    StaticChoiceMap(leaf_nodes, internal_nodes)
end

struct UsingWorld{V, Tr, n} <: Gen.GenerativeFunction{V, UsingWorldTrace}
    kernel::Gen.GenerativeFunction{V, Tr}
    addrs::NTuple{n, Symbol}
    memoized_gen_fns::NTuple{n, GenerativeFunction}
end
function UsingWorld(kernel::GenerativeFunction, addr_to_gen_fn...)
    addrs = Tuple([addr for (addr, gen_fn) in addr_to_gen_fn])
    gen_fns = Tuple([gen_fn for (addr, gen_fn) in addr_to_gen_fn])
    UsingWorld(kernel, addrs, gen_fns)
end

############
# generate #
############

(gen_fn::UsingWorld)(args...) = get_retval(simulate(gen_fn, args))
Gen.simulate(gen_fn::UsingWorld, args::Tuple) = generate(gen_fn, args)

function Gen.generate(gen_fn::UsingWorld, args::Tuple, constraints::ChoiceMap)
    world = World(gen_fn.addrs, gen_fn.memoized_gen_fns)
    begin_generate!(world, get_submap(constraints, :world))
    kernel_tr, kernel_weight = generate(gen_fn.kernel, (world, args...), get_submap(constraints, :kernel))
    world_weight = end_generate!(world)
    
    score = get_score(kernel_tr) + get_score(world)
    tr = UsingWorldTrace(kernel_tr, weight, score, args, gen_fn)
    weight = kernel_weight + world_weight
    
    (tr, weight)
end

###########
# updates #
###########

# TODO: update
# TODO: regenerate


# TODO: gradients?