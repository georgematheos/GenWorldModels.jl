#########
# Trace #
#########

struct UsingWorldTrace{V, Tr} <: Gen.Trace
    kernel_tr::Tr
    world::World
    score::Float64
    args::Tuple
    gen_fn::GenerativeFunction{V, UsingWorldTrace{V, Tr}}
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

struct UsingWorld{V, Tr, n} <: Gen.GenerativeFunction{V, UsingWorldTrace{V, Tr}}
    kernel::Gen.GenerativeFunction{V, Tr}
    addrs::NTuple{n, Symbol}
    memoized_gen_fns::NTuple{n, GenerativeFunction}
end
function UsingWorld(kernel::GenerativeFunction, addr_to_gen_fn::Vararg{Pair{Symbol, <:GenerativeFunction}})
    addrs = Tuple([addr for (addr, gen_fn) in addr_to_gen_fn])
    @assert all(addrs .!= :kernel) ":kernel may not be a memoized generative function address"
    gen_fns = Tuple([gen_fn for (addr, gen_fn) in addr_to_gen_fn])
    UsingWorld(kernel, addrs, gen_fns)
end

function Base.getindex(tr::UsingWorldTrace, addr::Pair)
    first, second = addr
    if first == :kernel
        return tr.kernel_tr[second]
    elseif first == :world
        mgf_addr, rest = second
        if rest isa Pair
            key, rest = rest
            return get_trace(tr.world, Call(mgf_addr, key))[rest]
        else
            return get_trace(tr.world, Call(mgf_addr, rest))[]
        end
    else
        error("Invalid address")
    end
end

############
# generate #
############

function count_appearances_of_value_deep(choicemap::ChoiceMap, addr, value)
    count = 0
    if has_value(choicemap, addr) && choicemap[addr] == value
        count += 1
    end
    for (_, submap) in get_submaps_shallow(choicemap)
        count += count_appearances_of_value_deep(submap, addr, value)
    end
    return count
end

"""
    check_world_counts_agree_with_generate_choicemaps(world, kernel_tr)

Throws an error if the `world`'s tracking of how many times each address was looked up
differs from the number of times each address would need to be looked up according to the kernel
trace, and the traces for the decisions stored in `world` for the memoized generative functions
invoked by the kernel.
"""
function check_world_counts_agree_with_generate_choicemaps(world, kernel_tr)
    core_error_msg(addr, key, expected_count, actual_counts) = """
    The world's tracking expected $(addr => key) to be looked up 
    $expected_count times in the kernel, but the kernel's choicemap
    states that it was only looked up $actual_counts times.
    """
    kernel_choices = get_choices(kernel_tr)
    for (mgf_addr, mgf_submap) in get_submaps_shallow(get_choices(world))
        for (lookup_key, submap) in get_submaps_shallow(mgf_submap)
            num_expected_kernel_lookups = get_number_of_expected_kernel_lookups(world.lookup_counts, mgf_addr => lookup_key)
            num_actual_kernel_lookups = count_appearances_of_value_deep(kernel_choices, 
                metadata_addr(world) => mgf_addr, lookup_key
            )

            if num_expected_kernel_lookups < num_actual_kernel_lookups
                # I don't see how this could ever occur, but might as well check for it
                # but note to future users: if you are triggering this error you are doing something very whacky
                error(core_error_msg(mgf_addr, lookup_key, num_expected_kernel_lookups, num_actual_kernel_lookups))
            elseif num_expected_kernel_lookups > num_actual_kernel_lookups
                error(core_error_msg(mgf_addr, lookup_key, num_expected_kernel_lookups, num_actual_kernel_lookups) * """
                One possible explaination for this issue is that there are untraced
                calls to `lookup_or_generate` in the kernel.  Another possible explaination is that for some reason
                a traced call to `lookup_or_generate` is resulting in multiple calls to the `lookup_or_generate` function
                even though it is only adding a single instance of this call to the choicemap.
                """)
            end

        end
    end
end

# (gen_fn::UsingWorld)(args...) = get_retval(simulate(gen_fn, args))
# Gen.simulate(gen_fn::UsingWorld, args::Tuple) = generate(gen_fn, args)

function Gen.generate(gen_fn::UsingWorld, args::Tuple, constraints::ChoiceMap; check_proper_usage=true)
    world = World(gen_fn.addrs, gen_fn.memoized_gen_fns)
    begin_generate!(world, get_submap(constraints, :world))
    kernel_tr, kernel_weight = generate(gen_fn.kernel, (world, args...), get_submap(constraints, :kernel))
    world_weight = end_generate!(world)
    
    score = get_score(kernel_tr) + total_score(world)
    tr = UsingWorldTrace(kernel_tr, world, score, args, gen_fn)
    weight = kernel_weight + world_weight
    
    if check_proper_usage
        check_world_counts_agree_with_generate_choicemaps(world, kernel_tr)
    end

    (tr, weight)
end

###########
# updates #
###########

# TODO: these kwargs won't usually propagate through multiple update calls as is
function Gen.update(tr::UsingWorldTrace, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap; check_no_constrained_calls_deleted=true)
    world = World(tr.world) # shallow copy of the world object

    # tell the world we are doing an update, and have it update all the values
    # for the constraints for values it has already generated
    world_diff = begin_update!(world, get_submap(constraints, :world))

    (new_kernel_tr, kernel_weight, kernel_retdiff, kernel_discard) = update(
        tr.kernel_tr, (world, args...), (world_diff, argdiffs...), get_submap(constraints, :kernel)
    )

    world_weight, world_discard = end_update!(world, kernel_discard, check_no_constrained_calls_deleted)

    new_score = get_score(new_kernel_tr) + total_score(world)
    new_tr = UsingWorldTrace(new_kernel_tr, world, new_score, args, tr.gen_fn)
    weight = kernel_weight + world_weight
    discard = StaticChoiceMap(NamedTuple(), (world=world_discard, kernel=kernel_discard))

    (new_tr, weight, kernel_retdiff, discard)
end

# TODO: regenerate

# TODO: gradients?