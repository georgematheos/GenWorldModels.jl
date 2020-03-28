#################
# Structs, etc. #
#################

"""
    MemoizedGenFnSpec(gen_fn::GenerativeFunction, address::Symbol, recursive_dependencies)
    
A spec for a memoized generative function to be used in a `UsingMemoized` combinator.
- `gen_fn` is the generative function to memoize.
- `address` is the address in the `UsingMemoized` combinator which will store the choices from `gen_fn`; this should not be `:kernel`
- `dependencies`: a tuple of addresses for all the other memoized generative functions which should be passed as arguments to `gen_fn`
"""
struct MemoizedGenFnSpec
    gen_fn::GenerativeFunction
    address::Symbol
    dependencies::NTuple{n, Symbol} where n
end

"""
    MemoizedGenFnSpec(addr => gen_fn)
    MemoizedGenFnSpec(addr => (gen_fn, dep1_addr, dep2_addr, ..., dep_n_addr))
    
Syntactic sugar for constructing a `MemoizedGenFnSpec`.
Takes in a pair with the address as the "key",
and with the "value" being either
1. The generative function to be memoized. In this case the MemoizedGenFnSpec has will have no dependencies.
or
2. A tuple where the first element is the generative function to be memoized, and the remaining elements are the
addresses of the dependencies for this MemoizedGenFnSpec.
"""
function MemoizedGenFnSpec(p::Pair{Symbol, <:GenerativeFunction})
    addr, gen_fn = p
    MemoizedGenFnSpec(gen_fn, addr, ())
end

function MemoizedGenFnSpec(p::Pair{Symbol, Tuple{<:GenerativeFunction, Vararg{Symbol}}})
    addr = p[1]
    gen_fn = p[2][1]
    deps = p[2][2:end]
    MemoizedGenFnSpec(gen_fn, addr, deps)
end

struct UsingMemoizedTrace{V, Tr} <: Gen.Trace
    kernel_tr::Tr
    memoized_gen_fns::Vector{<:MemoizedGenFn}
    score::Float64
    args::Tuple
    gen_fn::GenerativeFunction{V, UsingMemoizedTrace{V, Tr}}
end

"""
    UsingMemoized(kernel::GenerativeFunction, memoized_gen_fn_specs::Vararg{MemoizedGenFnSpec})
    
A combinator which runs the `kernel` generative function in such a way that
it can use generative functions specified in `memoized_gen_fn_specs` as if they are memoized.
"""
# TODO: document more fully, including invariants/contract with kernel.
struct UsingMemoized{V, Tr} <: Gen.GenerativeFunction{V, UsingMemoizedTrace{V, Tr}}
    kernel::Gen.GenerativeFunction{V, Tr}
    gen_fn_specs::Vector{MemoizedGenFnSpec}
    addr_to_idx::Dict{Symbol, Int64}
    
    function UsingMemoized{V, Tr}(kernel, gen_fn_specs::Vararg{MemoizedGenFnSpec}) where {V, Tr}
        addr_to_idx = Dict(spec.address => i for (i, spec) in enumerate(gen_fn_specs))
        collected_addrs = collect(keys(addr_to_idx))
        
        @assert unique(collected_addrs) == collected_addrs "Memoized gen fn addresses must be unique"
        @assert !in(:kernel, collected_addrs) "`:kernel` may not be a memoized gen fn address"
        for spec in gen_fn_specs
            @assert all(addr in collected_addrs for addr in spec.dependencies) "dependency addresses for $(spec.address) are not all known memoized gen fn addresses."
        end
        new{V, Tr}(kernel, collect(gen_fn_specs), addr_to_idx)
    end
end

UsingMemoized(kernel::GenerativeFunction{V, Tr}, gen_fn_specs...) where {V, Tr} = UsingMemoized{V, Tr}(kernel, gen_fn_specs...)
# syntactic sugar constructor
function UsingMemoized(kernel::GenerativeFunction{V, Tr}, gen_fn_specs::Vararg{<:Pair}) where {V, Tr}
    gen_fn_specs = [MemoizedGenFnSpec(spec) for spec in gen_fn_specs]
    UsingMemoized(kernel, gen_fn_specs...)
end

# eventually we may give users the option to create a `UsingMemoized` function
# which specifies not to check the validity of constraints, for performance,
# but for now, always do these checks
# ie. on `generate`, check that every constraint for the memoized functions are for
# an index which actually got used
do_check_constraint_validity(::UsingMemoized) = true

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
    for (addr, i) in get_gen_fn(tr).addr_to_idx
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

"""
    set_mgf_dependencies!(mgfs::Vector{MemoizedGenFn}, gen_fn::UsingMemoized)
    
Tell every mgf in `mgfs` what its dependencies are, assuming `mgfs` is indexed
in the same order as `gen_fn.gen_fn_specs`.
"""
function set_mgf_dependencies!(mgfs::Vector{<:MemoizedGenFn}, gen_fn::UsingMemoized)
    for (mgf, spec) in zip(mgfs, gen_fn.gen_fn_specs)
        dep_indices = (gen_fn.addr_to_idx[addr] for addr in spec.dependencies)
        deps = [mgfs[i] for i in dep_indices]
        __set_dependencies__!(mgf, deps)
    end
end

function prepare_mgfs(gen_fn, constraints)
    # for every function to be memoized, create a `MemoizedGenFn` lookup table.
    memoized_gen_fns = [MemoizedGenFn(spec.gen_fn) for spec in gen_fn.gen_fn_specs]

    # tell each `MemoizedGenFn` all it's dependency mgfs
    set_mgf_dependencies!(memoized_gen_fns, gen_fn)
    
    # pre-generate a value for each MGF for each value given in the `constraints`
    mgf_generate_weight = 0.
    for (addr, submap) in get_submaps_shallow(constraints)
        if addr == :kernel
            continue
        end
        mgf = memoized_gen_fns[gen_fn.addr_to_idx[addr]]
        
        for (idx, gen_constraints) in get_submaps_shallow(submap)
            mgf_generate_weight += __generate_value__!(mgf, idx, gen_constraints)
        end
    end

    return (memoized_gen_fns, mgf_generate_weight)
end

function Gen.generate(gen_fn::UsingMemoized{V, Tr}, args::Tuple, constraints::ChoiceMap) where {V, Tr}
    memoized_gen_fns, mgf_generate_weight = prepare_mgfs(gen_fn, constraints)
    
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
    set_mgf_dependencies!(new_memoized_gen_fns, get_gen_fn(tr))
    
    mgf_diffs::Vector{Union{CopiedMemoizedGenFnDiff, UpdatedIndicesDiff}} = [CopiedMemoizedGenFnDiff() for _=1:length(new_memoized_gen_fns)]
    
    discard = choicemap()
    total_weight = 0.
    
    for (addr, idx_to_constraint_map) in get_submaps_shallow(constraints)
        if addr == :kernel
            continue
        end
        
        mgf_discard = choicemap()
        mgf_idx = get_gen_fn(tr).addr_to_idx[addr]
        mgf = new_memoized_gen_fns[mgf_idx]
        updated_indices_to_retdiffs = Dict{Any, Diff}()
        
        for (idx, constraints_for_mgf) in get_submaps_shallow(idx_to_constraint_map)
            if __has_value__(mgf, idx)
                weight, retdiff, dsc = __update__!(mgf, idx, constraints_for_mgf)
                set_submap!(mgf_discard, idx, dsc)
                updated_indices_to_retdiffs[idx] = retdiff
            else
                weight = __generate_value__!(mgf, idx, constraints_for_mgf)
            end
            total_weight += weight
        end
        
        if !isempty(mgf_discard)
            set_submap!(discard, addr, mgf_discard)
        end
        
        if length(updated_indices_to_retdiffs) > 0
            mgf_diffs[mgf_idx] = UpdatedIndicesDiff(updated_indices_to_retdiffs)
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
    discarded_total_score = update_lookup_counts!(new_memoized_gen_fns, discard, tr.gen_fn.addr_to_idx, kernel_discard)
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
    for (addr, i) in tr.gen_fn.addr_to_idx
        choices = __get_choices__(tr.memoized_gen_fns[i])
        set_submap!(cm, addr, choices)
    end
    return cm
end

# TODO: gradient tracking?
Gen.has_argument_grads(tr::UsingMemoizedTrace) = map(_ -> false, get_args(tr))
Gen.accepts_output_grad(tr::UsingMemoizedTrace) = false