abstract type LookupTable{K, V} end

"""
    __lookup__!(l::LookupTable, idx)
Returns the index `idx` from the LookupTable. For internal use only.
"""
function __lookup__!(l::LookupTable)
    error("Not implemented")
end

# a lookup table which already has all its values filled in
struct ConcreteLookupTable{K, V} <: LookupTable{K, V}
    table::Dict{K, V}
end
__lookup__!(c::ConcreteLookupTable, idx) = getindex(c.table, idx)
    
mutable struct MemoizedGenFn{Tr <: Gen.Trace, K, V} <: LookupTable{K, V} 
    gen_fn::GenerativeFunction{V, Tr}
    subtraces::PersistentHashMap{K, Tr}
    lookup_counts::PersistentHashMap{K, Int}
    total_score::Float64
    unique_idx_lookup_addr::Symbol
    weight_tracker::Float64
    dependencies::Vector{MemoizedGenFn}
end

function MemoizedGenFn{Tr, K, V}(gen_fn, subtraces, lookup_counts, total_score, unique_idx_lookup_addr) where {Tr, K, V}
    MemoizedGenFn{Tr, K, V}(gen_fn, subtraces, lookup_counts, total_score, unique_idx_lookup_addr, 0., [])
end

function Base.copy(mgf::MemoizedGenFn{Tr, K, V}) where {Tr, K, V}
    MemoizedGenFn{Tr, K, V}(mgf.gen_fn, mgf.subtraces, mgf.lookup_counts, mgf.total_score, mgf.unique_idx_lookup_addr, mgf.weight_tracker, mgf.dependencies)
end

# constructor which decides it's own `unique_idx_lookup_addr`
function MemoizedGenFn{Tr, K, V}(gen_fn, subtraces, lookup_counts, total_score)  where {Tr <: Gen.Trace, K, V}
    # generate a unique symbol for each MGF to track its lookups with
    # TODO: If I optionally passed in the `addr` this would be referring to,
    # I could add that to the symbol to make it more recognizable for the user
    MemoizedGenFn{Tr, K, V}(gen_fn, subtraces, lookup_counts, total_score, gensym("__LOOKED_UP_IDX__"))
end

# empty memoized gen fn with no decisions yet
# by default set K = Any
function MemoizedGenFn(gen_fn::GenerativeFunction{V, Tr}) where {V, Tr}
    MemoizedGenFn{Tr, Any, V}(gen_fn, PersistentHashMap{Any, Tr}(), PersistentHashMap{Any, Int}(), 0.)
end

__idx_lookup_addr__(l::ConcreteLookupTable) = :__CONCRETE_LT_LOOKED_UP_IDX__
__idx_lookup_addr__(l::MemoizedGenFn) = l.unique_idx_lookup_addr

"""
    __set_dependencies__!(mgf::MemoizedGenFn, deps::Vector{<:MemoizedGenFn})
    
Tells `mgf` which MGFs are it's dependencies.  This must be called
before generating any values in the mgf since the dependencies
are passed into the underyling generative function as arguments.

`__set_dependencies__!` may be called at anytime to overwrite the current
dependencies with the new ones passed in.
"""
function __set_dependencies__!(mgf::MemoizedGenFn, deps::Vector{<:MemoizedGenFn})
    mgf.dependencies = deps
end

"""
    __zero_weight_tracker__!
    
Every time a new value is generated, the weight from this `generate` call is
added to `mgf.weight_tracker`.  This method zeros this tracker so we can
start tracking all weights generated starting from a certain point in program
execution.
"""
function __zero_weight_tracker__!(mgf::MemoizedGenFn)
    mgf.weight_tracker = 0.
end

"""
    __get_tracked_weight__

Returns the weight tracked by the mgf.
"""
function __get_tracked_weight__(mgf::MemoizedGenFn)
    mgf.weight_tracker
end

"""
    __lookup__!(mgf::MemoizedGenFn{Tr, K, V}, idx::L) where {Tr, K, V, L<:K}
    
Looks up `idx` in `mgf`.  If no value exists for the index, generates one.
Increments the lookup count for `idx` in `mgf`.
"""
function __lookup__!(mgf::MemoizedGenFn{Tr, K, V}, idx::L) where {Tr, K, V, L<:K}
    if !haskey(mgf.subtraces, idx)
        # TODO: if there is nonaddressed randomness, we may need to track the weight from this
        weight = __generate_value__!(mgf::MemoizedGenFn, idx)
    end
    mgf.lookup_counts = assoc(mgf.lookup_counts, idx, mgf.lookup_counts[idx] + 1)
    return get_retval(mgf.subtraces[idx])
end

"""
    __generate_value__!(mgf::MemoizedGenFn, idx, constraints::ChoiceMap)
    
This generates a value for the given `idx`, using the given
`constraints` in the `generate` call on the generative function.
Adds the weight from this generate call to `mgf.weight_tracker`,
and updates the total score of the `mgf` according to the score
of this new lookup.
"""
function __generate_value__!(mgf::MemoizedGenFn, idx, constraints::ChoiceMap)
    @assert !haskey(mgf.subtraces, idx)
    @assert !haskey(mgf.lookup_counts, idx)
    
    tr, weight = generate(mgf.gen_fn, (idx,), constraints)
    mgf.subtraces = assoc(mgf.subtraces, idx, tr)
    mgf.lookup_counts = assoc(mgf.lookup_counts, idx, 0)
    mgf.total_score += get_score(tr)
    
    mgf.weight_tracker += weight
    
    return weight
end

__generate_value__!(mgf::MemoizedGenFn, idx) = __generate_value__!(mgf::MemoizedGenFn, idx, EmptyChoiceMap())

__total_score__(mgf::MemoizedGenFn) = mgf.total_score

function __get_choices__(mgf::MemoizedGenFn)
    cm = choicemap()
    for (idx, tr) in mgf.subtraces
        set_submap!(cm, idx, get_choices(tr))
    end
    return cm
end

function __has_value__(mgf::MemoizedGenFn, idx)
    haskey(mgf.subtraces, idx)
end

function __update__!(mgf::MemoizedGenFn, idx, constraints::ChoiceMap)
    old_subtrace = mgf.subtraces[idx]
    new_subtrace, weight, retdiff, discard = update(old_subtrace, (idx,), (NoChange(),), constraints)
    mgf.total_score += get_score(new_subtrace) - get_score(old_subtrace)
    mgf.subtraces = assoc(mgf.subtraces, idx, new_subtrace)
    
    (weight, retdiff, discard)
end

function __decrease_count__!(mgf::MemoizedGenFn, idx, by_amount)
    mgf.lookup_counts = assoc(mgf.lookup_counts, idx, mgf.lookup_counts[idx] - by_amount)
    return mgf.lookup_counts[idx]
end
__decrease_count__!(_::ConcreteLookupTable, _, _) = nothing

function __delete_idx__!(mgf, idx)
    subtrace = mgf.subtraces[idx]
    mgf.subtraces = dissoc(mgf.subtraces, idx)
    mgf.lookup_counts = dissoc(mgf.lookup_counts, idx)
    mgf.total_score -= get_score(subtrace)
    return subtrace
end

abstract type MemoizedGenFnDiff <: Gen.Diff end

# this diff denotes that some of the values
# sampled in the MGF have changed; includes info
# of which indices has changed,
# and the retdiffs for these changed indices
# does not contain any info about if new indices have been added
struct UpdatedIndicesDiff <: MemoizedGenFnDiff
    updated_indices_to_retdiffs::Dict{Any, <:Diff}
end

# this diff denotes that a memoized gen function
# was copied (so is no longer equal to the original),
# but none of the internal information has changed
struct CopiedMemoizedGenFnDiff <: MemoizedGenFnDiff end