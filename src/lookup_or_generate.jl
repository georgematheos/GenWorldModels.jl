####################
# LookupOrGenerate #
####################

# TODO: could add some more information about the return type to the trace

struct LookupOrGenerateTrace <: Gen.Trace
    call::MemoizedGenerativeFunctionCall
    val
end

Gen.get_args(tr::LookupOrGenerateTrace) = (tr.call,)
Gen.get_retval(tr::LookupOrGenerateTrace) = tr.val
Gen.get_score(tr::LookupOrGenerateTrace) = 0.
Gen.get_gen_fn(tr::LookupOrGenerateTrace) = lookup_or_generate
Gen.project(tr::LookupOrGenerateTrace, selection::EmptySelection) = 0.
function Gen.get_choices(tr::LookupOrGenerateTrace)
    # TODO: static choicemap for performance?
    choicemap(
        (metadata_addr(tr.call.world) => addr(tr.call), key(tr.call))
    )
end

struct LookupOrGenerate <: GenerativeFunction{Any, LookupOrGenerateTrace} end
const lookup_or_generate = LookupOrGenerate()

@inline (gen_fn::LookupOrGenerate)(args...) = get_retval(simulate(gen_fn, args))

function Gen.generate(gen_fn::LookupOrGenerate, args::Tuple{MemoizedGenerativeFunctionCall{WorldType, addr}}, constraints::EmptyChoiceMap) where {WorldType, addr}
    mgf_call, = args
    val = lookup_or_generate!(mgf_call.world, Call(addr, mgf_call.key); reason_for_call=:generate)
    tr = LookupOrGenerateTrace(mgf_call, val)
    (tr, 0.)
end

# since this is a "delta dirac" to just lookup the value in the world, the simulate and generate
# distributions are the same. (however, the world may be in a simulate state, in which case
# the underlying calls to generate values in the world will be `simulate`; we don't need to 
# worry about that here, though)
@inline Gen.simulate(gen_fn::LookupOrGenerate, args::Tuple) = Gen.generate(gen_fn, args, EmptyChoiceMap())[1]

function Gen.generate(gen_fn::LookupOrGenerate, args::Tuple, constraints::ChoiceMap)
    if isempty(constraints)
        generate(gen_fn, args)
    else
        error("generate(lookup_or_generate, ...) should only be called with empty constraints")
    end
end

function Gen.propose(gen_fn::LookupOrGenerate, args::Tuple)
    mgf_call, = args
    retval = lookup_or_propose!(mgf_call.world, Call(addr(mgf_call), mgf_call.key))
    
    # we can return an EmptyChoiceMap for choices since we don't need to track
    # dependencies for a propose value, since the generated world is discarded afterwards.
    (EmptyChoiceMap(), 0., retval)
end

function Gen.assess(gen_fn::LookupOrGenerate, args::Tuple, ::EmptyChoiceMap)
    mgf_call, = args
    retval = lookup_or_assess!(mgf_call.world, Call(addr(mgf_call), mgf_call.key))

    (0., retval)
end

Gen.project(::LookupOrGenerateTrace, ::Selection) = 0.

##########
# Update #
##########

function Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap, ::Selection)
    error("lookup_or_generate may not be updated with constraints.")
end

# handle "regenerate"ing the trace
Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple, ::Selection, ::Selection) = update(tr, args, argdiffs, EmptyChoiceMap(), EmptySelection())

Gen.update(tr::LookupOrGenerateTrace, ::Tuple, ::Tuple{NoChange}, ::EmptyChoiceMap, ::Selection) = (tr, 0., NoChange(), EmptyChoiceMap())

# key change
function Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{MGFCallKeyChangeDiff}, ::EmptyChoiceMap, ::Selection)
    mgf_call = args[1]

    # run a full update/generate cycle in the world for this call
    new_val = lookup_or_generate!(mgf_call.world, Call(addr(mgf_call), key(mgf_call)); reason_for_call=:key_change)
    
    # the key looked up is the only thing exposed in the choicemap; this has changed so the whole old choicemap is the discard
    discard = get_choices(tr)
    
    new_tr = LookupOrGenerateTrace(mgf_call, new_val)
    retdiff = new_val == get_retval(tr) ? NoChange() : UnknownChange()
    (new_tr, 0., retdiff, discard)
end

# to-be-updated
function Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{ToBeUpdatedDiff}, ::EmptyChoiceMap, ::Selection)
    mgf_call = args[1]
    # run a full update/generate cycle in the world for this call
    new_val = lookup_or_generate!(mgf_call.world, Call(addr(mgf_call), key(mgf_call)); reason_for_call=:to_be_updated)
    new_tr = LookupOrGenerateTrace(mgf_call, new_val)
    retdiff = new_val == get_retval(tr) ? NoChange() : UnknownChange()
    (new_tr, 0., retdiff, EmptyChoiceMap())
end

# value has changed, but we don't need to update, and haven't changed key
function Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{MGFCallValChangeDiff}, ::EmptyChoiceMap, ::Selection)
    valdiff = argdiffs[1].diff
    new_call = args[1]
    new_val = get_val(new_call.world, call(new_call))
    new_tr = LookupOrGenerateTrace(new_call, new_val)
    (new_tr, 0., valdiff, EmptyChoiceMap())
end

# UnknownChange - this is likely caused by use of a dynamic generative function which doesn't propagate diffs
# we need to figure out which of the other update functions we should dispatch to
# this function just replicates the diff propagation behavior implemented above in the Base.getindex methods
function Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{UnknownChange}, ::EmptyChoiceMap, s::Selection)
    new_call = args[1]
    if key(tr.call) != key(new_call)
        diff = MGFCallKeyChangeDiff()
    else
        diff = WorldUpdateDiff(new_call.world)[addr(new_call)][key(new_call)]
        if diff != NoChange() && diff != ToBeUpdatedDiff()
            diff = MGFCallValChangeDiff(diff)
        end
    end
    return Gen.update(tr, args, (diff,), EmptyChoiceMap(), s)
end

# TODO: gradients