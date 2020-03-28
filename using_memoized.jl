#################
# Structs, etc. #
#################

struct UsingMemoizedTrace{V, Tr} <: Gen.Trace
    kernel_tr::Tr
    memoized_gen_fns::Vector{<:MemoizedGenFn}
    score::Float64
    args::Tuple
    gen_fn::GenerativeFunction{V, UsingMemoizedTrace{V, Tr}}
end

struct UsingMemoized{V, Tr} <: Gen.GenerativeFunction{V, UsingMemoizedTrace{V, Tr}}
    kernel::Gen.GenerativeFunction{V, Tr}
    gen_fns::Vector{GenerativeFunction}
    addr_to_gen_fn_idx::Dict{Symbol, Int64}
    
    function UsingMemoized{V, Tr}(kernel, addr_to_gen_fn_list...) where {V, Tr}
        gen_fns = [fn for (addr, fn) in addr_to_gen_fn_list]
        addr_to_gen_fn_idx = Dict(addr => i for (i, (addr, fn)) in enumerate(addr_to_gen_fn_list))
        
        @assert all(addr != :kernel for addr in keys(addr_to_gen_fn_idx)) "`:kernel` may not be a memoized gen fn address"
        
        new{V, Tr}(kernel, gen_fns, addr_to_gen_fn_idx)
    end
end

# eventually we may give users the option to create a `UsingMemoized` function
# which specifies not to check the validity of constraints, for performance,
# but for now, always do these checks
# ie. on `generate`, check that every constraint for the memoized functions are for
# an index which actually got used
do_check_constraint_validity(::UsingMemoized) = true

function UsingMemoized(kernel::GenerativeFunction{V, Tr}, addr_to_gen_fn_list...) where {V, Tr}
    UsingMemoized{V, Tr}(kernel, addr_to_gen_fn_list...)
end

####################
# Helper functions #
####################

function merge_in_counts!(counts, cnts)
    for (idx, cnt) in cnts
        counts[idx] = get(counts, idx, 0) + cnt
    end
end

"""
    get_lookup_counts(choices::ChoiceMap, lookup_table::LookupTable)
    
Returns a dictionary from every index `choices` indicates we looked up in `lookup_table`
during the execution of a kernel function which produced the `choices` choicemap
to the number of times this `choices` choicemap suggests the index was looked up.
"""
function get_lookup_counts(choices::ChoiceMap, lookup_table::LookupTable)
    counts = Dict{Any, Int}()
    if has_value(choices, __idx_lookup_addr__(lookup_table))
        idx = choices[__idx_lookup_addr__(lookup_table)]
        counts[idx] = 1
    end
    for (_, submap) in get_submaps_shallow(choices)
        cnts = get_lookup_counts(submap, lookup_table)
        merge_in_counts!(counts, cnts)
    end
    return counts
end

"""
    check_constraint_validity(tr::UsingMemoizedTrace, constraints::ChoiceMap)

Ensures that the every constraint on memoized gen function values provided in `constraints`
is for an index which ended up getting looked up somewhere in the kernel in `tr`.
"""
function check_constraint_validity(tr::UsingMemoizedTrace, constraints::ChoiceMap)
    for (addr, i) in get_gen_fn(tr).addr_to_gen_fn_idx
        looked_up_inds = keys(get_lookup_counts(get_choices(tr.kernel_tr), tr.memoized_gen_fns[i]))
        for (idx, submap) in get_submaps_shallow(get_submap(constraints, addr))
            if !in(idx, looked_up_inds)
                error("Constraints provided a value for $addr => $idx, which was never looked up in resulting trace.")
            end
        end
    end
end

###############
# GFI methods #
###############

### GENERATE ###

@inline (gen_fn::UsingMemoized)(args...) = Gen.simulate(gen_fn, args)
@inline Gen.simulate(gen_fn::UsingMemoized, args::Tuple) = Gen.get_retval(Gen.generate(gen_fn, args)[1])
@inline Gen.generate(gen_fn::UsingMemoized, args::Tuple) = Gen.generate(gen_fn, args, EmptyChoiceMap())

function Gen.generate(gen_fn::UsingMemoized{V, Tr}, args::Tuple, constraints::ChoiceMap) where {V, Tr}
    # for every function to be memoized, create a `MemoizedGenFn` lookup table.
    # pass in the constraints for the given address to each one to pre-populate
    # the table with the values given in `constraints`
    memoized_gen_fns = Vector{MemoizedGenFn}(undef, length(gen_fn.gen_fns))
    mgf_generate_weight = 0.
    
    for (addr, i) in gen_fn.addr_to_gen_fn_idx
        mgf, weight = generate_memoized_gen_fn_with_prepopulated_indices(gen_fn.gen_fns[i], get_submap(constraints, addr))
        memoized_gen_fns[i] = mgf
        mgf_generate_weight += weight
    end
        
    kernel_args = (memoized_gen_fns..., args...)
    kernel_tr, kernel_weight = generate(gen_fn.kernel, kernel_args, get_submap(constraints, :kernel))
    
    score = sum(__total_score__(mgf) for mgf in memoized_gen_fns) + get_score(kernel_tr)
    
    tr = UsingMemoizedTrace{V, Tr}(kernel_tr, memoized_gen_fns, score, args, gen_fn)
    
    if do_check_constraint_validity(gen_fn)
        check_constraint_validity(tr, constraints)
    end
    
    return tr, kernel_weight + mgf_generate_weight
end

### UPDATE ###

"""
    update_lookup_table_values(tr, constraints)

This creates a shallow copy of each memoized lookup table, and if `constraints` specifies values for some indices,
modifies these indices in the new lookup tables.  Tracks the change in total score
induced by this update, as well as the total weight for all these updates.  Tracks the `diffs` for
the MGFs, and the `discard` for the overall `UsingMemoizedTrace` update coming from these updates.

Returns `(score_delta, total_weight, new_memoized_gen_fns, mgf_diffs, discard)`.
"""
function update_lookup_table_values(tr, constraints)
    new_memoized_gen_fns = [copy(mgf) for mgf in tr.memoized_gen_fns]
    mgf_diffs::Vector{Union{CopiedMemoizedGenFnDiff, UpdatedIndicesDiff}} = [CopiedMemoizedGenFnDiff() for _=1:length(new_memoized_gen_fns)]
    
    discard = choicemap()
    total_weight = 0.
    
    for (addr, i) in get_gen_fn(tr).addr_to_gen_fn_idx
        mgf_discard = choicemap()
        if get_submap(constraints, addr) == EmptyChoiceMap()
            continue
        end
        
        updated_indices_to_retdiffs = Dict{Any, Diff}()
        mgf = new_memoized_gen_fns[i]
        
        for (idx, submap) in get_submaps_shallow(get_submap(constraints, addr))
            if __has_value__(mgf, idx)
                weight, retdiff, dsc = __update__!(mgf, idx, submap)
                set_submap!(mgf_discard, idx, dsc)
                updated_indices_to_retdiffs[idx] = retdiff
            else
                weight = __generate_value__!(mgf, idx, submap)
            end
            total_weight += weight
        end
        
        if !isempty(mgf_discard)
            set_submap!(discard, addr, mgf_discard)
        end
        
        if length(updated_indices_to_retdiffs) > 0
            mgf_diffs[i] = UpdatedIndicesDiff(updated_indices_to_retdiffs)
        end
    end
    
    (total_weight, new_memoized_gen_fns, mgf_diffs, discard)
end

function update_lookup_counts!(mgf::MemoizedGenFn, kernel_discard::ChoiceMap)
    counts = get_lookup_counts(kernel_discard, mgf)
    discarded_choicemap = choicemap()
    total_discarded_subtrace_score = 0.
    for (idx, count) in counts
        new_count = __decrease_count__!(mgf, idx, count)
        @assert new_count >= 0
        if new_count == 0
            discarded_subtrace = __delete_idx__!(mgf, idx)
            total_discarded_subtrace_score += get_score(discarded_subtrace)
            set_submap!(discarded_choicemap, idx, get_choices(discarded_subtrace))
        end
    end
    return (discarded_choicemap, total_discarded_subtrace_score)
end

function update_lookup_counts!(memoized_gen_fns::AbstractVector{<:MemoizedGenFn}, discard_to_update::DynamicChoiceMap, addr_to_idx::Dict{Symbol, Int}, kernel_discard::ChoiceMap)
    discarded_indices = Dict{Symbol, Any}()
    total_discarded_subtrace_score = 0.
    for (addr, i) in addr_to_idx
        mgf = memoized_gen_fns[i]
        disc_choicemap, disc_score = update_lookup_counts!(mgf, kernel_discard)
        set_submap!(discard_to_update, addr, merge(disc_choicemap, get_submap(discard_to_update, addr)))
        total_discarded_subtrace_score += disc_score
    end
    return total_discarded_subtrace_score
end

function Gen.update(tr::UsingMemoizedTrace{V, Tr}, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap) where {V, Tr}
    ### update values in lookup table given in `constraints` ###
    # the return value `new_memoized_gen_fns` is a complete copy 
    (total_weight, new_memoized_gen_fns, mgf_diffs, discard) = update_lookup_table_values(tr, constraints)

    ### update kernel ###
    kernel_args = (new_memoized_gen_fns..., args...)
    kernel_argdiffs = (mgf_diffs..., argdiffs...)
    
    # we want to track the total weight of all new values generated during
    # the `kernel` update, so zero the weight trackers on the mgfs
    for mgf in new_memoized_gen_fns
        __zero_weight_tracker__!(mgf)
    end
    
    new_kernel_tr, weight, retdiff, kernel_discard = update(tr.kernel_tr, kernel_args, kernel_argdiffs, get_submap(constraints, :kernel))
    set_submap!(discard, :kernel, kernel_discard)

    total_weight += weight
    total_tracked_weight = sum(__get_tracked_weight__(mgf) for mgf in new_memoized_gen_fns)
    total_weight += sum(__get_tracked_weight__(mgf) for mgf in new_memoized_gen_fns)
    
    ### update the counts in the lookup tables and remove choices for indices which now have 0 lookups ###
    # this will also update the `discard` with the dropped subtrace choicemaps
    discarded_total_score = update_lookup_counts!(new_memoized_gen_fns, discard, tr.gen_fn.addr_to_gen_fn_idx, kernel_discard)
    total_weight -= discarded_total_score
    
    ### create trace ###
    kernel_score = get_score(new_kernel_tr)
    mgf_score = sum(__total_score__(mgf) for mgf in new_memoized_gen_fns)
    score = kernel_score + mgf_score
    
    new_tr = UsingMemoizedTrace(new_kernel_tr, new_memoized_gen_fns, score, args, tr.gen_fn)
    
    # sanity check: did we remove any values which were also passed in in `constraints`?
    # TODO: could we make this faster using incremental computation?
    if do_check_constraint_validity(get_gen_fn(tr))
        check_constraint_validity(new_tr, constraints)
    end
    
    return (new_tr, total_weight, retdiff, discard)
end

function Gen.regenerate(tr::UsingMemoizedTrace{V, Tr}, args::Tuple, argdiffs::Tuple, selection::Selection) where {V, Tr}
    return
end

Gen.get_args(tr::UsingMemoizedTrace) = tr.args
Gen.get_retval(tr::UsingMemoizedTrace) = get_retval(tr.kernel_tr)
Gen.get_score(tr::UsingMemoizedTrace) = tr.score
Gen.get_gen_fn(tr::UsingMemoizedTrace) = tr.gen_fn
function Gen.project(tr::UsingMemoizedTrace, sel::Selection)
    error("NOT IMPLEMENTED")
end

function Gen.get_choices(tr::UsingMemoizedTrace)
    cm = choicemap()
    set_submap!(cm, :kernel, get_choices(tr.kernel_tr))
    for (addr, i) in tr.gen_fn.addr_to_gen_fn_idx
        choices = __get_choices__(tr.memoized_gen_fns[i])
        set_submap!(cm, addr, choices)
    end
    return cm
end

# TODO: gradient tracking?
Gen.has_argument_grads(tr::UsingMemoizedTrace) = map(_ -> false, get_args(tr))
Gen.accepts_output_grad(tr::UsingMemoizedTrace) = false