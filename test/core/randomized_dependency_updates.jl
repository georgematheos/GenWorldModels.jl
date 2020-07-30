using FunctionalCollections
#=
This file will generate some fairly random dependency structure,
and use both the UsingWorld combinator and the Unfold combinator
to generate & update such that the programs should have the same semantics.
The tests will check that UsingWorld matches the behavior of Unfold.
=#

function Base.reduce(op, itr::Diffed; init)
    reduced = reduce(op, strip_diff(itr); init=init)
    if get_diff(itr) == NoChange()
        Diffed(reduced, NoChange())
    else
        Diffed(reduced, UnknownChange())
    end
end

####################################
# Static UsingWorld implementation #
####################################

@gen (static, diffs) function num_vals(world, t::Tuple{})
    mean ~ lookup_or_generate(world[:args][:num_values_mean])
    num ~ poisson(mean)
    return num
end

@gen (static, diffs) function lookup_random_value(world, num_vals)::Float64
    lookup ~ uniform_discrete(1, num_vals)
    looked_up_val ~ lookup_or_generate(world[:vals][lookup])
    return looked_up_val
end
lookup_random_values = Map(lookup_random_value)

# for this, we need to constrain that `num_lookups` is 0 for `n=1` so we don't have infinite recursion
# I don't want to put an if statement here, else I couldn't use the static DSL
@gen (static, diffs) function sample_val_static(world, n)
    num_lookups ~ poisson(2)
    num_vals ~ lookup_or_generate(world[:num_vals][()])
    vals ~ lookup_random_values(fill(world, num_lookups), fill(num_vals, num_lookups))
    total = reduce(+, vals; init=0.)
    val ~ normal(total, 1)
    return val
end

@gen (static, diffs) function lookup_val(world, n)
    val ~ lookup_or_generate(world[:vals][n])
    return val
end

@gen (static, diffs) function sample_vals_static_kernel(world, indices_to_sample)
    vals ~ Map(lookup_val)(fill(world, length(indices_to_sample)), indices_to_sample)
    return vals
end

sample_vals_world = UsingWorld(sample_vals_static_kernel, :num_vals => num_vals, :vals => sample_val_static; world_args=(:num_values_mean,))
@load_generated_functions()

##########################
# Dynamic implementation #
##########################

# initial state should be an empty PersistentHashMap
@gen function lookup_or_sample_val(n, state, num_vals)
    if n in keys(state)
        return (state[n], state)
    end

    new_state = state
    total = 0

    num_lookups ~ poisson(2)
    for i=1:num_lookups
        lookup = {:lookup_idx => i} ~ uniform_discrete(1, num_vals)
        (looked_up_val, new_state) = {:looked_up_val => i} ~ lookup_or_sample_val(lookup, new_state, num_vals)
        total += looked_up_val
    end
    val ~ normal(total, 1)
    new_state = assoc(new_state, n, val)
    return (val, new_state)
end

@gen function sample_vals_dynamic(indices_to_sample)
    if !isempty(indices_to_sample)
        num_vals ~ poisson(150)
    end
    state = PersistentHashMap()
    vals = []
    for index in indices_to_sample
        (new_val, state) = {:vals => index} ~ lookup_or_sample_val(index, state, num_vals)
        push!(vals, new_val)
    end
    return vals
end

##########################################
# Randomly generate dependency structure #
##########################################

#=
Returns `deps`, a vector representing the dependency structure
we generated.
`deps[i]` = a vector of the dependencies call `i` should have.
Sets call 1 to look up nothing and then generates a DAG from that.
=#
@gen function generate_random_dependency_structure(N)
    generated = [1]
    deps = Vector{Vector}(undef, N)
    deps[1] = []
    for i=2:N
        num_deps = {:num_deps => i} ~ poisson(2)
        dep = [
            generated[{:dep => i => j} ~ uniform_discrete(1, length(generated))]
            for j=1:num_deps
        ]
        deps[i] = dep
        push!(generated, i)
    end
    return deps
end

function append_pair(prefix::Pair, rest)
    (fst, scnd) = prefix
    fst => append_pair(scnd, rest)
end
append_pair(prefix, rest) = prefix => rest

function generate_world_trace(indices_to_sample, deps, vals, num_nodes)
    using_world_constraints = choicemap()
    using_world_constraints[:world => :num_vals => () => :num] = num_nodes
    for (i, dep) in enumerate(deps)
        using_world_constraints[:world => :vals => i => :num_lookups] = length(dep)
        for (j, idx_to_lookup) in enumerate(dep)
            using_world_constraints[:world => :vals => i => :vals => j => :lookup] = idx_to_lookup
        end
        using_world_constraints[:world => :vals => i => :val] = vals[i]
    end

    # we are constraining all the calls, but we might not generate all of them, so turn off `check_all_constraints_used`
    return generate(sample_vals_world, (150, indices_to_sample,), using_world_constraints; check_all_constraints_used=false)
end

function generate_dynamic_trace(indices_to_sample, deps, vals, num_nodes)
    dynamic_constraints = choicemap()
    dynamic_constraints[:num_vals] = num_nodes
    constraints_added_for = []
    function add_constraints_for_idx(idx, prefix)
        if idx in constraints_added_for
            return
        end
        push!(constraints_added_for, idx)
        dep = deps[idx]
        dynamic_constraints[append_pair(prefix, :num_lookups)] = length(dep)
        for (j, idx_to_lookup) in enumerate(dep)
            dynamic_constraints[append_pair(prefix, :lookup_idx => j)] = idx_to_lookup
            if !(idx_to_lookup in constraints_added_for)
                add_constraints_for_idx(idx_to_lookup, append_pair(prefix, :looked_up_val => j))
            end
        end
        dynamic_constraints[append_pair(prefix, :val)] = vals[idx]
    end
    for idx in indices_to_sample
        add_constraints_for_idx(idx, :vals => idx)
    end

    return generate(sample_vals_dynamic, (indices_to_sample,), dynamic_constraints)    
end

function generate_random_traces_and_run_tests()
    num_nodes = poisson(150)
    # first, generate a random dependency structure
    dep_struct_trace = simulate(generate_random_dependency_structure, (num_nodes,))
    deps = get_retval(dep_struct_trace)

    num_indices_to_sample = poisson(4)
    indices_to_sample = unique([uniform_discrete(1, num_nodes) for _=1:num_indices_to_sample])
    vals = [normal(0, 1) for _=1:num_nodes]

    using_world_trace, using_world_weight = generate_world_trace(indices_to_sample, deps, vals, num_nodes)
    dynamic_trace, dynamic_weight = generate_dynamic_trace(indices_to_sample, deps, vals, num_nodes)

    @test using_world_weight ≈ dynamic_weight
    @test get_score(using_world_trace) ≈ get_score(dynamic_trace)
    @test get_retval(using_world_trace) == get_retval(dynamic_trace)

    return (dep_struct_trace, using_world_trace, dynamic_trace, vals)
end

function generate_updated_dep_struct(dep_struct_trace, new_num_vals)
    PROB_OF_DEP_CHANGE = 0.35

    old_deps = get_retval(dep_struct_trace)
    addrs = []
    selection = DynamicSelection()
    for (i, dep) in enumerate(old_deps)
        if bernoulli(PROB_OF_DEP_CHANGE)
            push!(selection, :num_deps => i)
        end
        for (j, _) in enumerate(dep)
            if bernoulli(PROB_OF_DEP_CHANGE)
                push!(selection, :dep => i => j)
            end
        end
    end
    (new_dep_struct_trace, _, _) = regenerate(dep_struct_trace, (new_num_vals,), (UnknownChange(),), selection)

    return new_dep_struct_trace
end

function generate_new_vals(new_num_vals, old_vals)
    PROB_OF_VAL_CHANGE = 0.35

    new_vals = Vector{Float64}(undef, new_num_vals)
    for i=1:min(length(old_vals), new_num_vals)
        old_val = old_vals[i]
        if bernoulli(PROB_OF_VAL_CHANGE)
            new_vals[i] = normal(0, 1)
        else
            new_vals[i] = old_val
        end
    end
    for i in (length(old_vals) + 1):new_num_vals
        new_vals[i] = normal(0, 1)
    end
    return new_vals
end

function generate_new_indices_to_lookup(new_num_vals, old_indices_to_lookup)
    PROB_OF_IDX_CHANGE = 0.35
    new_num_indices_to_sample = poisson(4)
    new_indices = Vector{Int}(undef, new_num_indices_to_sample)
    for i=1:new_num_indices_to_sample
        if i > length(old_indices_to_lookup) || bernoulli(PROB_OF_IDX_CHANGE) || old_indices_to_lookup[i] > new_num_vals
            new_indices[i] = uniform_discrete(1, new_num_vals)
        else
            new_indices[i] = old_indices_to_lookup[i]
        end
    end
    return unique(new_indices)
end

function update_world_trace(old_world_trace, new_deps, new_vals, new_indices_to_sample)
    old_choices = get_choices(old_world_trace)
    old_num_indices_sampled = length(get_retval(old_world_trace))
    old_num_vals = old_num_indices_sampled == 0 ? 0 : old_choices[:world => :num_vals => () => :num]
    constraints = choicemap()

    if !has_value(old_choices, :world => :num_vals => () => :num) ||  old_choices[:world => :num_vals => () => :num] != length(new_vals)
        constraints[:world => :num_vals => () => :num] = length(new_vals)
    end

    for i in 1:length(new_vals)
        dep = new_deps[i]
        old_val_map = get_submap(old_choices, :world => :vals => i)
        if !isempty(old_val_map)
            if old_val_map[:num_lookups] != length(dep)
                constraints[:world => :vals => i => :num_lookups] = length(dep)
            end
            for (j, idx_to_lookup) in enumerate(dep)
                if j > old_val_map[:num_lookups] || old_val_map[:vals => j => :lookup] != idx_to_lookup
                    constraints[:world => :vals => i => :vals => j => :lookup] = idx_to_lookup
                end
            end
            if old_val_map[:val] != new_vals[i]
                constraints[:world => :vals => i => :val] = new_vals[i]
            end
        else
            constraints[:world => :vals => i => :num_lookups] = length(dep)
            for (j, idx_to_lookup) in enumerate(dep)
                constraints[:world => :vals => i => :vals => j => :lookup] = idx_to_lookup
            end
            constraints[:world => :vals => i => :val] = new_vals[i]    
        end
    end

    new_tr, weight, retdiff, discard = update(old_world_trace, (150, new_indices_to_sample,), (NoChange(), UnknownChange(),), constraints, Gen.AllSelection(); check_no_constrained_calls_deleted=false)
    return (new_tr, weight)
end

function perform_random_update_and_run_tests(dep_struct_trace, using_world_trace, dynamic_trace, old_vals; verbose=false)
    new_num_vals = poisson(150)
    new_dep_struct_trace = generate_updated_dep_struct(dep_struct_trace, new_num_vals)
    new_vals = generate_new_vals(new_num_vals, old_vals)
    new_indices_to_sample = generate_new_indices_to_lookup(new_num_vals, get_args(using_world_trace)[1])
    new_deps = get_retval(new_dep_struct_trace)

    (new_using_world_trace, using_world_weight) = update_world_trace(using_world_trace, new_deps, new_vals, new_indices_to_sample)

    # there is no point in updating the dynamic trace rather than generating a new one,
    # since the weight will not match the world model anyway, since the addresses of different choices
    # change when we change the dependency structure, so we have to constrain more choices
    # here than we do for the world model.  so for simplicity I just generate a new trace
    (new_dynamic_trace, dynamic_weight) = generate_dynamic_trace(new_indices_to_sample, new_deps, new_vals, new_num_vals)

    if !isapprox(get_score(new_using_world_trace), get_score(new_dynamic_trace); atol=1e-10)
        println("SCORES UNEQUAL")
        println("world trace score:", get_score(new_using_world_trace))
        println("dynamic trace score: ", get_score(new_dynamic_trace))

        println("old indices to sample:")
        println(get_args(using_world_trace)[1])
        println("new indices to sample:")
        println(new_indices_to_sample)
        println("old dep structure: ", get_retval(dep_struct_trace))
        println("new dep structure: ", get_retval(new_dep_struct_trace))
        changed_vals = [i for i=1:new_num_vals if i > length(old_vals) || old_vals[i] != new_vals[i]]
        println("Changed vals: $changed_vals")

        dep_struct = [
            i => [
                new_using_world_trace[:world => :vals => i => :vals => j => :lookup]
                for j=1:new_using_world_trace[:world => :vals => i => :num_lookups]
            ] for i=1:new_num_vals if !isempty(get_submap(get_choices(new_using_world_trace), :world => :vals => i))
        ]
        println("observed dep struct:")
        println(dep_struct)

        ord_new = Vector(undef, new_using_world_trace.world.call_sort.max_index)
        for (call, i) in new_using_world_trace.world.call_sort.call_to_idx
            ord_new[i] = call
        end
        println("new call sort: ")
        println(ord_new)

        ord_old = Vector(undef, using_world_trace.world.call_sort.max_index)
        for (call, i) in using_world_trace.world.call_sort.call_to_idx
            ord_old[i] = call
        end
        println("old call sort: ")
        println(ord_old)

        println("OLD TRACE SCORES:")
        for (call, tr) in GenWorldModels.all_traces(using_world_trace.world.traces)
            println("$call score: ", get_score(tr))
        end

        gend_world_trace, _ = generate_world_trace(new_indices_to_sample, new_deps, new_vals, new_num_vals)

        println("DIFFERENCE IN TRACE SCORES BETWEEN GENERATED AND UPDATED:")
        tot_diff = 0.
        for (call, gend_tr) in GenWorldModels.all_traces(gend_world_trace.world.traces)
            upd_tr = GenWorldModels.get_trace(new_using_world_trace.world.traces, call)
            gscore = get_score(gend_tr)
            uscore = get_score(upd_tr)
            if !isapprox(gscore, uscore )
                tot_diff += gscore - uscore
                println("DIFF FOR $call : Generated=$gscore, Updated=$uscore")
            end
        end
        println("total difference in Gen'd and Up'd scores is $tot_diff")

        # println("new world trace:")
        # display(get_choices(new_using_world_trace))
        # println("new dynamic trace:")
        # display(get_choices(new_dynamic_trace))
        error()
    end

    @test isapprox(get_score(new_using_world_trace), get_score(new_dynamic_trace); atol=1e-10)
    @test get_retval(new_using_world_trace) == get_retval(new_dynamic_trace)
    @test isapprox(using_world_weight, get_score(new_using_world_trace) - get_score(using_world_trace); atol=1e-10)
    @test isapprox(using_world_weight, get_score(new_dynamic_trace) - get_score(dynamic_trace); atol=1e-10)

    return (new_dep_struct_trace, new_using_world_trace, new_dynamic_trace, new_vals)
end

@testset "randomized dependency updates" begin
    NUM_CYCLES_TO_RUN = 5
    NUM_UPDATES_PER_CYCLE = 10
    for cycle in 1:NUM_CYCLES_TO_RUN
       (dep_struct_trace, using_world_trace, dynamic_trace, vals) = generate_random_traces_and_run_tests()
        for update in 1:NUM_UPDATES_PER_CYCLE
            (dep_struct_trace, using_world_trace, dynamic_trace, vals) = perform_random_update_and_run_tests(dep_struct_trace, using_world_trace, dynamic_trace, vals; verbose=(cycle==2&&update==3))
        end
    end
end