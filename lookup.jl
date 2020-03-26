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
    val = __getindex__(lookup_table, idx)
    
    tr = LookupTrace(lookup_table, idx, val)
    
    (tr, 0.)
end

function Gen.generate(gen_fn::Lookup, args::Tuple, constraints::ChoiceMap)
    error("generate(lookup, ...) should only be called with empty constraints, since the choices are entirely deterministic")
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