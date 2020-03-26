struct LookupTrace <: Gen.Trace
    lookup_table::LookupTable
    idx
    val
end

struct Lookup <: GenerativeFunction{Any, LookupTrace} end
const lookup = Lookup()

@inline (gen_fn::Lookup)(args...) = Gen.simulate(gen_fn, args)
@inline Gen.simulate(gen_fn::Lookup, args::Tuple) = Gen.get_retval(Gen.generate(gen_fn, args)[1])

function Gen.generate(gen_fn::Lookup, args::Tuple, constraints::EmptyChoiceMap)
    lookup_table, idx = args
    val = __lookup__!(lookup_table, idx)
    
    tr = LookupTrace(lookup_table, idx, val)
    
    (tr, 0.)
end

function Gen.generate(gen_fn::Lookup, args::Tuple, constraints::ChoiceMap)
    error("generate(lookup, ...) should only be called with empty constraints, since the choices are entirely deterministic")
end

function Gen.update(tr::LookupTrace, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap)
    error("lookup may not be updated with constraints since it is a deterministic gen function")
end

function Gen.update(tr::LookupTrace, args::Tuple, argdiffs::Tuple{NoChange, NoChange}, constraints::EmptyChoiceMap)
    (tr, 0., NoChange(), EmptyChoiceMap())
end

function Gen.update(tr::LookupTrace, args::Tuple, argdiffs::Tuple{<:Diff, NoChange}, constraints::EmptyChoiceMap)
    lookup_table, idx = args
    lt_diff = argdiffs[1]
    if !in(idx, keys(lt_diff.updated_indices_to_retdiffs))
        return (tr, 0., NoChange(), EmptyChoiceMap())
    end
    
    new_tr = LookupTrace(lookup_table, idx, __lookup__!(lookup_table, idx))
    
    # above, we called __lookup__! in order to get what the value has been changed to,
    # but this corresponds to a lookup which has already been counted in the lookup counts,
    # so we need to subtract off this lookup from the counts
    __decrease_count__!(new_tr.lookup_table, idx, 1)
    
    return (new_tr, 0., UnknownChange(), choicemap((:val, tr.val)))
end

function Gen.update(tr::LookupTrace, args::Tuple, argdiffs::Tuple{<:Diff, <:Diff}, constraints::EmptyChoiceMap)
    lookup_table, idx = args
    
    # this will add 1 to the lookup counts; this is the correct behavior since this may correspond
    # to looking up a new index
    # the `UsingMemoized` update will handle decreasing the counts which are no longer made
    # by scanning the discard choicemaps
    new_tr = LookupTrace(lookup_table, idx, __lookup__!(lookup_table, idx))
    
    return (new_tr, 0., UnknownChange(), get_choices(tr))
end

Gen.get_args(tr::LookupTrace) = (tr.lookup_table, idx)
Gen.get_retval(tr::LookupTrace) = tr.val
Gen.get_score(tr::LookupTrace) = 0.
Gen.get_gen_fn(tr::LookupTrace) = lookup
Gen.project(tr::LookupTrace, selection::EmptySelection) = 0.
function Gen.get_choices(tr::LookupTrace)
    choicemap(
        (:val, tr.val),
        # have some internally-known address which stores the index we just looked up in the choicemap
        (__idx_lookup_addr__(tr.lookup_table), tr.idx)
    )
end

# TODO: we can probably do grad stuff
Gen.has_argument_grads(tr::LookupTrace) = map(_ -> false, get_args(trace))
Gen.accepts_output_grad(tr::LookupTrace) = false