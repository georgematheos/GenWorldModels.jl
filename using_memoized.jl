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

function MemoizedGenFnSpec(p::Pair{Symbol, <:Tuple{<:GenerativeFunction, Vararg{Symbol}}})
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
    sort::Dict{CallTo, Int}
end

"""
    UsingMemoized(kernel::GenerativeFunction, memoized_gen_fn_specs::Vararg{MemoizedGenFnSpec})
    
A combinator which runs the `kernel` generative function in such a way that
it can use generative functions specified in `memoized_gen_fn_specs` as if they are memoized.

The `kernel` should be a generative function whose first `n` inputs are
`LookupTable` objects, where `n = length(memoized_gen_fn_specs)`.
We guarentee that when calling any GFI method of `kernel`, these lookup tables
will behave identically to `ConcreteLookupTable` objects which are pre-populated
with values for every index the `kernel` looks up in them via traced calls to
the `lookup` gen function.  We make no guarantees about correctness if these
lookup tables are used in ways other than looking up objects in traced `lookup`
calls.

The returned `UsingMemoized()` function will behave identically to if
it first determined all values the `kernel` needs access to, then generates these
values according to the `memoized_gen_fn_specs` and puts them in `ConcreteLookupTable`s,
then calls the `kernel` on these. On updates,
it will behave as if it first determines the new set of all values it needs to have in the lookup
tables, then populates these and only these, keeping old values when possible,
then calls update on `kernel` with these new lookup tables.

To generate a value at index `idx` for the lookup table corresponding to
`mgf_spec = memoized_gen_fn_specs[i]`, we call `mgf_spec.gen_fn(idx, deps...)`.
Here, the first parameter is the index we are looking up, and `deps` is a list of lookup
tables for every "dependency" specified in `mgf_spec.dependencies`.  `mgf_spec.gen_fn`
may treat these as `ConcreteLookupTable`s which are pre-populated with every value needed
to correctly generate the value for `idx`, so long as it only uses them in traced `lookup` calls,
and so long as there are not cyclic dependencies among indices.
(Eg. we cannot have `lookup(lookup_table, 1)` depend on the value of `lookup(lookup_table, 2)`
and simultaneously have `lookup(lookup_table, 2)` depend on the value of `lookup(lookup_table, 1)`,
since there is no way to calculate this!)

The choicemap will have address `:kernel` for the choicemap of the `kernel` function.
The choicemap for `mgf_spec.gen_fn` called on index `idx` will be at address
`mgf_spec.address => idx` in the returned choicemap.
"""
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
    for (key, submap) in get_submaps_shallow(choices)
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
        looked_up_inds = keys(get_lookup_counts(get_choices(tr), tr.memoized_gen_fns[i]))
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
    call_stack = Stack{CallTo}()
    # for every function to be memoized, create a `MemoizedGenFn` lookup table.
    memoized_gen_fns = [MemoizedGenFn(spec.gen_fn, call_stack, i, spec.address) for (i, spec) in enumerate(gen_fn.gen_fn_specs)]

    # tell each `MemoizedGenFn` all its dependency mgfs
    set_mgf_dependencies!(memoized_gen_fns, gen_fn)

    # pre-generate a value for each MGF for each value given in the `constraints`
    for (addr, submap) in get_submaps_shallow(constraints)
        if addr == :kernel
            continue
        end
        mgf = memoized_gen_fns[gen_fn.addr_to_idx[addr]]

        for (idx, gen_constraints) in get_submaps_shallow(submap)
            __generate_value__!(mgf, idx, gen_constraints)
        end
    end

    return memoized_gen_fns
end

"""
    topologically_sort_lookups(mgfs::Vector{<:MemoizedGenFn})
    
Topologically sorts the calls from these `mgfs`.
Returns a Dictionary `sort` such that
if CallTo1 must be calculated before CallTo2
because CallTo2 looks up the value in CallTo1 to calculate it,
then `sort[CallTo1] < sort[CallTo2]`.
"""
# TODO: TEST THIS FUNCTION INDIVIDUALLY!!!
function topologically_sort_lookups(mgfs::Vector{<:MemoizedGenFn})
    relevant_calls = union([Set(__get_all_calls__(mgf)) for mgf in mgfs]...)
    
    sort = Dict{CallTo, Int}()
    current_val = 0
    
    function visit(call::CallTo)
        if call in keys(sort)
            return
        end
        for call_this_was_looked_up_in in mgfs[call.mgf_idx].looked_up_in[call.idx]
            visit(call_this_was_looked_up_in)
        end
        
        sort[call] = current_val
        delete!(relevant_calls, call)
        current_val += 1
    end
    
    for call in relevant_calls
        visit(call)
    end
    
    return sort
end

function Gen.generate(gen_fn::UsingMemoized{V, Tr}, args::Tuple, constraints::ChoiceMap) where {V, Tr}
    memoized_gen_fns = prepare_mgfs(gen_fn, constraints)
    
    kernel_args = (memoized_gen_fns..., args...)
    kernel_tr, kernel_weight = generate(gen_fn.kernel, kernel_args, get_submap(constraints, :kernel))
    
    mgf_generate_weight = sum(__get_tracked_weight__(mgf) for mgf in memoized_gen_fns)
    
    # topologically sort all the `CallTo`s; get a dict s.t.
    # top_sort[CallTo1] < top_sort[CallTo2] if we must generate CallTo1
    # before CallTo2 (ie. generating CallTo2 involves a lookup of CallTo1)
    top_sort = topologically_sort_lookups(memoized_gen_fns)
    
    score = sum(__total_score__(mgf) for mgf in memoized_gen_fns) + get_score(kernel_tr)
    
    tr = UsingMemoizedTrace{V, Tr}(kernel_tr, memoized_gen_fns, score, args, gen_fn, top_sort)
    
    if do_check_constraint_validity(gen_fn)
        check_constraint_validity(tr, constraints)
    end
    
    return tr, kernel_weight + mgf_generate_weight
end

### UPDATE ###

"""
    update_lookup_table_values(tr, constraints)

This creates a shallow copy of each memoized lookup table, and if `constraints` specifies values for some indices,
modifies these indices in the new lookup tables.  Also updates all other calls to the memoized generative functions
which include lookups to the mgfs the `constraints` modifies as needed to maintain correctness.
Tracks the change in total score induced by this update, as well as the total weight for all the `update` calls.
Does not track the weight for `generate` calls; these should be tracked by the MGF weight trackers.
Tracks the `diffs` for the MGFs, and the `discard` for the overall `UsingMemoizedTrace` update coming from these updates.

Returns `(score_delta, total_weight_from_updates, new_memoized_gen_fns, mgf_diffs, discard)`.

Algorithm involves loading all updates we will do into a PriorityQueue, so runtime is
O(n*log(n) + runtime of update functions) where n=number of (mgf, idx) updates we need to do.
"""
function update_lookup_table_values!(new_memoized_gen_fns, tr, constraints)    
    mgf_diffs::Vector{Union{CopiedMemoizedGenFnDiff, UpdatedIndicesDiff}} = [CopiedMemoizedGenFnDiff() for _=1:length(new_memoized_gen_fns)]
    
    discard = choicemap()
    updates_weight = 0.
    
    pq = PriorityQueue{CallTo, Int}()
    given_constraints = Dict{CallTo, ChoiceMap}()
    
    for (addr, idx_to_constraint_map) in get_submaps_shallow(constraints)
        if addr == :kernel
            continue
        end
        
        mgf_idx = get_gen_fn(tr).addr_to_idx[addr]
        mgf = new_memoized_gen_fns[mgf_idx]

        for (idx, constraints_for_mgf) in get_submaps_shallow(idx_to_constraint_map)
            call = CallTo(mgf, idx)
            given_constraints[call] = constraints_for_mgf
            priority = call in keys(tr.sort) ? tr.sort[call] : typemax(Int)
            enqueue!(pq, call, priority)
        end
    end
    
    while !isempty(pq)
        call = dequeue!(pq)
        constraints = get(given_constraints, call, EmptyChoiceMap())
        mgf = new_memoized_gen_fns[call.mgf_idx]
        idx = call.idx
        
        if __has_value__(mgf, idx)
            dependency_diffs = [mgf_diffs[__get_mgf_index__(dep)] for dep in __get_dependencies__(mgf)]
            weight, retdiff, dsc = __update__!(mgf, idx, dependency_diffs, constraints)
            
            updates_weight += weight
            set_submap!(discard, __get_mgf_address__(mgf) => idx, dsc)
            
            if retdiff != NoChange()
                # note the diffs!
                i = __get_mgf_index__(mgf)
                if mgf_diffs[i] == CopiedMemoizedGenFnDiff()
                    mgf_diffs[i] = UpdatedIndicesDiff(Dict())
                end
                mgf_diffs[i].updated_indices_to_retdiffs[idx] = retdiff
                
                # we need to run update on every gen function this Call is looked up in
                for call_to in mgf.looked_up_in[idx]
                    # if this is in a `looked_up_in`, we should certainly have already sorted in this
                    # so no need to worry about the case there !(call_to in keys(tr.sort))
                    enqueue!(pq, call_to, tr.sort[call_to])
                end
            end
        else
            # if this value hasn't been generated, all there is to do is generate it
            # DO NOT ADD THE WEIGHT TO `updates_weight`; the generate weight is being tracked by the mgf's weight tracker
            __generate_value__!(new_memoized_gen_fns[call.mgf_idx], call.idx, constraints)
        end
    end
    
    (updates_weight, new_memoized_gen_fns, mgf_diffs, discard)
end

"""
    update_lookup_counts!(memoized_gen_fns::AbstractVector{<:MemoizedGenFn}, discard_to_update::DynamicChoiceMap, discard_to_scan::ChoiceMap)
    
Scans the `discard_to_scan` choicemap to see how many lookups for each index for each mgf in `memoized_gen_fns` were dropped.
Decreases the counts tracked in the `mgf`s accordingly.  If a count reaches zero, this deletes the index from
the MGF and tracks the score of the corresponding subtrace, then scans this subtrace to decrease counts
for lookups in the subtrace.

Returns the sum of the scores of all the discarded subtraces,
and updates `discard_to_update` so that `discard_to_update[mgf_address => idx]` will have the choicemap
for the subtrace for idx in mgf if this subtrace was dropped.
"""
function update_lookup_counts!(memoized_gen_fns::AbstractVector{<:MemoizedGenFn}, discard_to_update::DynamicChoiceMap, discard_to_scan::ChoiceMap)
    counts = [get_lookup_counts(discard_to_scan, mgf) for mgf in memoized_gen_fns]
    total_discarded_subtrace_score = 0.
    for (i, mgf) in enumerate(memoized_gen_fns)
        addr = __get_mgf_address__(mgf)
        for (idx, count) in counts[i]
            new_count = __decrease_count__!(mgf, idx, count)
            @assert new_count >= 0
            if new_count == 0
                discarded_subtrace = __delete_idx__!(mgf, idx)
                set_submap!(discard_to_update, addr => idx, get_choices(discarded_subtrace))
                total_discarded_subtrace_score += get_score(discarded_subtrace)
                
                # scan this subtrace we are dropping to see if we can decrease counts for the lookups needed to generate these values
                total_discarded_subtrace_score += update_lookup_counts!(memoized_gen_fns, discard_to_update, get_choices(discarded_subtrace))
            end
        end
    end
    
    return total_discarded_subtrace_score
end

"""
Create a shallow copy of each memoized generative function,
properly set dependencies, zero the weight trackers, and
give a new call stack object to each new mgf.
"""
function copy_mgfs_for_update(tr)
    call_stack = Stack{CallTo}()
    new_memoized_gen_fns = [MemoizedGenFn(mgf, call_stack) for mgf in tr.memoized_gen_fns]
    set_mgf_dependencies!(new_memoized_gen_fns, get_gen_fn(tr))
    # TODO: it looks like somehow the new counts are ending up on the old MGF array!!!

    # we want to track the total weight of all new values generated during
    # these updates, so zero the weight trackers on the mgfs
    for mgf in new_memoized_gen_fns
        __zero_weight_tracker__!(mgf)
    end
    
    return new_memoized_gen_fns
end

function Gen.update(tr::UsingMemoizedTrace{V, Tr}, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap) where {V, Tr}
    new_memoized_gen_fns = copy_mgfs_for_update(tr)
    
    ### update values in lookup table given in `constraints` ###
    # do these in topological order w.r.t. any recursive calls; update other lookups not in `constraints` if needed
    # do to other recursive calls which have been made
    (total_weight, new_memoized_gen_fns, mgf_diffs, discard) = update_lookup_table_values!(new_memoized_gen_fns, tr, constraints)
    
    ### update kernel ###
    kernel_args = (new_memoized_gen_fns..., args...)
    kernel_argdiffs = (mgf_diffs..., argdiffs...)
        
    new_kernel_tr, weight, retdiff, kernel_discard = update(tr.kernel_tr, kernel_args, kernel_argdiffs, get_submap(constraints, :kernel))
    set_submap!(discard, :kernel, kernel_discard)

    # add in the weight from the kernel call, and from all new generate calls
    total_weight += weight
    total_tracked_weight = sum(__get_tracked_weight__(mgf) for mgf in new_memoized_gen_fns)
    total_weight += total_tracked_weight
    
    ### update the counts in the lookup tables and remove choices for indices which should now have 0 lookups ###
    # this will also update the `discard` with the dropped subtrace choicemaps
    discarded_total_score = update_lookup_counts!(new_memoized_gen_fns, discard, discard)
    total_weight -= discarded_total_score
    
    ### create trace ###
    kernel_score = get_score(new_kernel_tr)
    mgf_score = sum(__total_score__(mgf) for mgf in new_memoized_gen_fns)
    score = kernel_score + mgf_score
    
    # TODO: dynamically update topological sort rather than redoing after each update
    new_sort = topologically_sort_lookups(new_memoized_gen_fns)
    new_tr = UsingMemoizedTrace(new_kernel_tr, new_memoized_gen_fns, score, args, tr.gen_fn, new_sort)
    
    # sanity check: did we remove any values which were also passed in in `constraints`?
    # TODO: could we make this faster using incremental computation?
    if do_check_constraint_validity(get_gen_fn(new_tr))
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
function Base.getindex(tr::UsingMemoizedTrace, addr::Symbol)
    if addr == :kernel
        return tr.kernel_tr[]
    else
        error("No value at address $addr")
    end
end
function Base.getindex(tr::UsingMemoizedTrace, addr::Pair)
    first, second = addr
    if first == :kernel
        return tr.kernel_tr[second]
    end
    
    i = get_gen_fn(tr).addr_to_idx[first]
    mgf = tr.memoized_gen_fns[i]
    if second isa Pair
        idx, query = second
        return mgf.subtraces[idx][query]
    else
        return mgf.subtraces[second][]
    end
end

# TODO: gradient tracking?
Gen.has_argument_grads(tr::UsingMemoizedTrace) = map(_ -> false, get_args(tr))
Gen.accepts_output_grad(tr::UsingMemoizedTrace) = false