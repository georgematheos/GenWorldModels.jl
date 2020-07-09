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

function Gen.get_choices(tr::UsingWorldTrace)
    full_choices = StaticChoiceMap((kernel=get_choices(tr.kernel_tr), world=get_choices(tr.world)))
    
    # return a choicemap which filters out all the choices addressed with `metadata_addr`,
    # since these are just used for internal tracking and should not be exposed
    AddressFilterChoiceMap(
        full_choices,
        addr -> addr != metadata_addr(tr.world)
    )
end

struct UsingWorld{num_world_args, num_mgfs, V, Tr} <: Gen.GenerativeFunction{V, UsingWorldTrace{V, Tr}}
    kernel::Gen.GenerativeFunction{V, Tr}
    mgf_addrs::NTuple{num_mgfs, Symbol}
    memoized_gen_fns::NTuple{num_mgfs, GenerativeFunction}
    world_arg_addrs::NTuple{num_world_args, Symbol}
end
function UsingWorld(kernel::GenerativeFunction, addr_to_gen_fn::Vararg{Pair{Symbol, <:GenerativeFunction}}; world_args=())
    mgf_addrs = Tuple([addr for (addr, gen_fn) in addr_to_gen_fn])
    @assert all(mgf_addrs .!= :kernel) ":kernel may not be a memoized generative function address"
    gen_fns = Tuple([gen_fn for (addr, gen_fn) in addr_to_gen_fn])
    UsingWorld(kernel, mgf_addrs, gen_fns, world_args)
end

function Base.getindex(tr::UsingWorldTrace, addr::Pair)
    first, second = addr
    if first == :kernel
        return tr.kernel_tr[second]
    elseif first == :world

        # TODO: if there is only one call for a MGF, we should be able to just look up that address

        mgf_addr, rest = second
        try 
            if rest isa Pair
                key, remaining = rest
                return get_trace(tr.world, Call(mgf_addr, key))[remaining]
            else
                return get_trace(tr.world, Call(mgf_addr, rest))[]
            end
        catch
            key = rest isa Pair ? rest[1] : rest
            error("No lookup for $(mgf_addr => key) found in the world.")
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
    states that it was looked up $actual_counts times.
    """
    kernel_choices = get_choices(kernel_tr)
    for (mgf_addr, mgf_submap) in get_submaps_shallow(get_choices(world))
        for (lookup_key, submap) in get_submaps_shallow(mgf_submap)
            num_expected_kernel_lookups = get_number_of_expected_kernel_lookups(world.lookup_counts, Call(mgf_addr, lookup_key))
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


function extract_world_args(uw::UsingWorld{num_world_args}, args::Tuple) where {num_world_args}
    world_args = NamedTuple{uw.world_arg_addrs}(args[begin:num_world_args])
    nonworld_args = args[num_world_args+1:end]
    (world_args, nonworld_args)
end

function (gen_fn::UsingWorld)(args...)
    (_, _, retval) = propose(gen_fn, args)
    retval
end

function Gen.generate(gen_fn::UsingWorld, args::Tuple, constraints::ChoiceMap; check_proper_usage=true, check_all_constraints_used=true)
    world_args, args = extract_world_args(gen_fn, args)

    world = World(gen_fn.mgf_addrs, gen_fn.memoized_gen_fns, world_args)
    begin_generate!(world, get_submap(constraints, :world))
    kernel_tr, kernel_weight = generate(gen_fn.kernel, (world, args...), get_submap(constraints, :kernel))
    world_weight = end_generate!(world, check_all_constraints_used)
    
    score = get_score(kernel_tr) + total_score(world)
    tr = UsingWorldTrace(kernel_tr, world, score, args, gen_fn)
    weight = kernel_weight + world_weight
    
    if check_proper_usage
        check_world_counts_agree_with_generate_choicemaps(world, kernel_tr)
    end

    (tr, weight)
end

function Gen.simulate(gen_fn::UsingWorld, args::Tuple; check_proper_usage=true, check_all_constraints_used=true)
    world_args, args = extract_world_args(gen_fn, args)

    world = World(gen_fn.mgf_addrs, gen_fn.memoized_gen_fns, world_args)
    begin_simulate!(world)
    kernel_tr = simulate(gen_fn.kernel, (world, args...))
    end_simulate!(world, check_all_constraints_used)

    score = get_score(kernel_tr) + total_score(world)
    tr = UsingWorldTrace(kernel_tr, world, score, args, gen_fn)
    
    if check_proper_usage
        check_world_counts_agree_with_generate_choicemaps(world, kernel_tr)
    end

    tr
end

function Gen.propose(gen_fn::UsingWorld, args::Tuple)
    world_args, args = extract_world_args(gen_fn, args)

    world = World(gen_fn.mgf_addrs, gen_fn.memoized_gen_fns, world_args)
    begin_propose!(world)
    kernel_choices, kernel_weight, retval = propose(gen_fn.kernel, (world, args...))
    world_choices, world_weight = end_propose!(world)

    weight = kernel_weight + world_weight
    choices = StaticChoiceMap((kernel=kernel_choices, world=world_choices))

    # no need to address filter since we shouldn't be populating with the mgf_address on a propose call

    return (choices, weight, retval)
end

function Gen.assess(gen_fn::UsingWorld, args::Tuple, choices::ChoiceMap)
    world_args, args = extract_world_args(gen_fn, args)

    world = World(gen_fn.mgf_addrs, gen_fn.memoized_gen_fns, world_args)
    begin_assess!(world, get_submap(choices, :world))
    kernel_weight, retval = assess(gen_fn.kernel, (world, args...), get_submap(choices, :kernel))
    world_weight = end_assess!(world)

    (kernel_weight + world_weight, retval)
end

function Gen.project(trace::UsingWorldTrace, selection::Selection)
    kernel_weight = project(trace.kernel_tr, selection[:kernel])
    world_weight = project_(trace.world, selection[:world])
    return kernel_weight + world_weight
end

###########
# updates #
###########

# TODO: these kwargs won't usually propagate through multiple update calls as is
function Gen.update(tr::UsingWorldTrace, args::Tuple, argdiffs::Tuple,
    spec::Gen.UpdateSpec, externally_constrained_addrs::Selection;
    check_no_constrained_calls_deleted=true
)
    world = World(tr.world) # shallow copy of the world object

    world_args, args = extract_world_args(tr.gen_fn, args)
    world_argdiffs, argdiffs = extract_world_args(tr.gen_fn, argdiffs)

    # tell the world we are doing an update, and have it update all the values
    # for the constraints for values it has already generated
    world_diff = begin_update!(world, get_subtree(spec, :world), get_subtree(externally_constrained_addrs, :world), world_args, world_argdiffs)
    
    (new_kernel_tr, kernel_weight, kernel_retdiff, kernel_discard) = update(
        tr.kernel_tr, (world, args...), (world_diff, argdiffs...), get_subtree(spec, :kernel), get_subtree(externally_constrained_addrs, :kernel)
    )

    world_weight, world_discard = end_update!(world, kernel_discard, check_no_constrained_calls_deleted)

    new_score = get_score(new_kernel_tr) + total_score(world)
    new_tr = UsingWorldTrace(new_kernel_tr, world, new_score, args, tr.gen_fn)
    weight = kernel_weight + world_weight
    discard = StaticChoiceMap((world=world_discard, kernel=kernel_discard))
    discard = AddressFilterChoiceMap(discard, addr -> addr != metadata_addr(world))

    (new_tr, weight, kernel_retdiff, discard)
end

# TODO: regenerate

# TODO: gradients?