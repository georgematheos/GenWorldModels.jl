#################
# Structs, etc. #
#################

struct UsingMemoizedTrace{V, Tr} <: Gen.Trace
    kernel_tr::Tr
    memoized_gen_fns::Vector{MemoizedGenFn}
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
        
        new{V, Tr}(kernel, gen_fns, addr_to_gen_fn_idx)
    end
end

# eventually we may give users the option to create a `UsingMemoized` function
# which specifies not to check the validity of constraints, for performance,
# but for now, always do these checks
# ie. on `generate`, check that every constraint for the memoized functions are for
# an index which actually got used
check_constraint_validity(::UsingMemoized) = true

function UsingMemoized(kernel::GenerativeFunction{V, Tr}, addr_to_gen_fn_list...) where {V, Tr}
    UsingMemoized{V, Tr}(kernel, addr_to_gen_fn_list...)
end

####################
# Helper functions #
####################

"""
    get_looked_up_inds(choices::ChoiceMap, lookup_table::LookupTable)

Returns a set of all the indices `choices` indicates were looked up in `lookup_table` during
the execution of a kernel function which produced the `choices` choicemap.
"""
function get_looked_up_inds(choices::ChoiceMap, lookup_table::LookupTable)
    inds = Set()
    if has_value(choices, __idx_lookup_addr__(lookup_table))
        push!(inds, choices[__idx_lookup_addr__(lookup_table)])
    end
    for (_, submap) in get_submaps_shallow(choices)
        other_inds = get_looked_up_inds(submap, lookup_table)
        union!(inds, other_inds)
    end
    inds
end

"""
    check_generate_constraint_validity(tr::UsingMemoizedTrace, constraints::ChoiceMap)

Ensures that the every constraint on memoized gen function values provided in `constraints`
ended up getting looked up somewhere in the kernel in `tr`.
"""
function check_generate_constraint_validity(tr::UsingMemoizedTrace, constraints::ChoiceMap)
    for (addr, i) in get_gen_fn(tr).addr_to_gen_fn_idx
        looked_up_inds = get_looked_up_inds(get_choices(tr.kernel_tr), tr.memoized_gen_fns[i])
        for (idx, submap) in get_submaps_shallow(get_submap(constraints, addr))
            if !in(idx, looked_up_inds)
                error("Constraints provided a value for $addr => $idx, which was never looked up.")
            end
        end
    end
end

###############
# GFI methods #
###############

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
    kernel_tr, kernel_weight = generate(gen_fn.kernel, kernel_args, constraints)
    
    score = sum(__total_score__(mgf) for mgf in memoized_gen_fns) + get_score(kernel_tr)
    
    tr = UsingMemoizedTrace{V, Tr}(kernel_tr, memoized_gen_fns, score, args, gen_fn)
    
    if check_constraint_validity(gen_fn)
        check_generate_constraint_validity(tr, constraints)
    end
    
    return tr, kernel_weight + mgf_generate_weight
end

Gen.get_args(tr::UsingMemoizedTrace) = tr.args
Gen.get_retval(tr::UsingMemoizedTrace) = get_retval(tr.kernel_tr)
Gen.get_score(tr::UsingMemoizedTrace) = tr.score
Gen.get_gen_fn(tr::UsingMemoizedTrace) = tr.gen_fn
function Gen.project(tr::UsingMemoizedTrace, sel::Selection)
    error("NOT IMPLEMENTED")
end

function Gen.get_choices(tr::UsingMemoizedTrace)
    cm = DynamicChoiceMap(get_choices(tr.kernel_tr))
    for (addr, i) in tr.gen_fn.addr_to_gen_fn_idx
        choices = __get_choices__(tr.memoized_gen_fns[i])
        set_submap!(cm, addr, choices)
    end
    return cm
end

# TODO: gradient tracking?
Gen.has_argument_grads(tr::UsingMemoizedTrace) = map(_ -> false, get_args(tr))
Gen.accepts_output_grad(tr::UsingMemoizedTrace) = false