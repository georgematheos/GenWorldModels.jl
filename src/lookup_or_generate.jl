####################
# LookupOrGenerate #
####################

# TODO: could add some more information about the return type to the trace

abstract type LookupOrGenerateTrace <: Gen.Trace end
# gfi methods with the same implementation for all concrete types of `LookupOrGenerateTrace`
Gen.get_score(tr::LookupOrGenerateTrace) = 0.
Gen.get_gen_fn(tr::LookupOrGenerateTrace) = lookup_or_generate
Gen.project(::LookupOrGenerateTrace, ::EmptySelection) = 0.
Gen.project(::LookupOrGenerateTrace, ::Selection) = 0.

"""
    lookup_or_generate(world[address][key])

Look up the value for the call with argument `key` to memoized generative function
with name `address` in the world, or generate a value for it if none currently exists.
"""
struct LookupOrGenerate <: GenerativeFunction{Any, LookupOrGenerateTrace} end
const lookup_or_generate = LookupOrGenerate()

struct SimpleLookupOrGenerateTrace <: LookupOrGenerateTrace
    call::MemoizedGenerativeFunctionCall
    val
end
Gen.get_retval(tr::SimpleLookupOrGenerateTrace) = tr.val
function Gen.get_choices(tr::SimpleLookupOrGenerateTrace)
    # TODO: use static choicemap
    choicemap((metadata_addr(tr.call.world) => addr(tr.call), key(tr.call)))
end
Gen.get_args(tr::SimpleLookupOrGenerateTrace) = (tr.call,)

@gen (static, diffs) function idx_to_id_lookup_or_generate(mgf_call)
    idx_obj = key(mgf_call)
    type = oupm_type(idx_obj)
    wrld = world(mgf_call)
    id_form ~ lookup_or_generate(wrld[type][idx(idx_obj)])
    val ~ lookup_or_generate(wrld[addr(mgf_call)][id_form])
    return val 
end
@load_generated_functions()

struct AutomaticIdxToIdLookupOrGenerateTrace <: LookupOrGenerateTrace
    tr::Gen.get_trace_type(idx_to_id_lookup_or_generate)
end
Gen.get_retval(tr::AutomaticIdxToIdLookupOrGenerateTrace) = get_retval(tr.tr)
Gen.get_choices(tr::AutomaticIdxToIdLookupOrGenerateTrace) = get_choices(tr.tr)
Gen.get_args(tr::AutomaticIdxToIdLookupOrGenerateTrace) = get_args(tr.tr)

@inline (gen_fn::LookupOrGenerate)(args...) = get_retval(simulate(gen_fn, args))

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

################################
# IdxToID LookupOrGenerate GFI #
################################
@inline function Gen.generate(gen_fn::LookupOrGenerate, args::Tuple{MemoizedGenerativeFunctionCall{<:Any, <:Any, <:OUPMType{Int}}}, ::EmptyChoiceMap)
    tr, weight = generate(idx_to_id_lookup_or_generate, args, EmptyChoiceMap())
    (AutomaticIdxToIdLookupOrGenerateTrace(tr), weight)
end
@inline function Gen.propose(gen_fn::LookupOrGenerate, args::Tuple{MemoizedGenerativeFunctionCall{<:Any, <:Any, <:OUPMType{Int}}})
    propose(idx_to_id_lookup_or_generate, args)
end
@inline function Gen.assess(gen_fn::LookupOrGenerate, args::Tuple{MemoizedGenerativeFunctionCall{<:Any, <:Any, <:OUPMType{Int}}}, ::EmptyChoiceMap)
    assess(idx_to_id_lookup_or_generate, args, EmptyChoiceMap())
end
@inline function Gen.update(
    tr::AutomaticIdxToIdLookupOrGenerateTrace,
    args::Tuple{MemoizedGenerativeFunctionCall},
    argdiffs::Tuple{<:Gen.Diff}, spec::Gen.UpdateSpec, ext_const_addrs::Gen.Selection
)
    println("updating LOG from $(get_args(tr)[1]) to $(args[1]) : diff is $(argdiffs[1])")
    new_tr, weight, retdiff, discard = update(tr.tr, args, argdiffs, spec, ext_const_addrs)
    (AutomaticIdxToIdLookupOrGenerateTrace(new_tr), weight, retdiff, discard)
end

###############################
# Simple LookupOrGenerate GFI #
###############################

function Gen.generate(gen_fn::LookupOrGenerate, args::Tuple{MemoizedGenerativeFunctionCall{<:Any, addr, <:Any}}, ::EmptyChoiceMap) where {addr}
    mgf_call, = args
    val = lookup_or_generate!(mgf_call.world, Call(addr, mgf_call.key); reason_for_call=:generate)
    tr = SimpleLookupOrGenerateTrace(mgf_call, val)
    (tr, 0.)
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

##################################
# Simple LookupOrGenerate Update #
##################################

function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap, ::Selection)
    error("lookup_or_generate may not be updated with constraints.")
end

# handle "regenerate"ing the trace
Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, argdiffs::Tuple, ::Selection, ::Selection) = update(tr, args, argdiffs, EmptyChoiceMap(), EmptySelection())
Gen.update(tr::SimpleLookupOrGenerateTrace, ::Tuple, ::Tuple{NoChange}, ::EmptyChoiceMap, ::Selection) = (tr, 0., NoChange(), EmptyChoiceMap())

### SimpleLookupOrGenerateTrace updates ###
# key change
function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{MGFCallKeyChangeDiff}, ::EmptyChoiceMap, ::Selection)
    mgf_call, = args

    # run a full update/generate cycle in the world for this call
    new_val = lookup_or_generate!(mgf_call.world, Call(addr(mgf_call), key(mgf_call)); reason_for_call=:key_change)
    
    # the key looked up is the only thing exposed in the choicemap; this has changed so the whole old choicemap is the discard
    discard = get_choices(tr)
    
    new_tr = SimpleLookupOrGenerateTrace(mgf_call, new_val)
    retdiff = new_val == get_retval(tr) ? NoChange() : UnknownChange()
    (new_tr, 0., retdiff, discard)
end

# to-be-updated
function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{ToBeUpdatedDiff}, ::EmptyChoiceMap, ::Selection)
    mgf_call = args[1]
    # run a full update/generate cycle in the world for this call
    new_val = lookup_or_generate!(mgf_call.world, Call(addr(mgf_call), key(mgf_call)); reason_for_call=:to_be_updated)
    new_tr = SimpleLookupOrGenerateTrace(mgf_call, new_val)
    retdiff = new_val == get_retval(tr) ? NoChange() : UnknownChange()
    (new_tr, 0., retdiff, EmptyChoiceMap())
end

# value has changed, but we don't need to update, and haven't changed key
function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{MGFCallValChangeDiff}, ::EmptyChoiceMap, ::Selection)
    valdiff = argdiffs[1].diff
    new_call = args[1]
    new_val = get_val(new_call.world, call(new_call))
    new_tr = SimpleLookupOrGenerateTrace(new_call, new_val)
    (new_tr, 0., valdiff, EmptyChoiceMap())
end

### UnknownChange dispatching ###
# UnknownChange  is likely caused by use of a dynamic generative function which doesn't propagate diffs
# we need to figure out which of the other update functions we should dispatch to
# this function just replicates the diff propagation behavior implemented above in the Base.getindex methods
@inline function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple{MemoizedGenerativeFunctionCall}, argdiffs::Tuple{UnknownChange}, ::EmptyChoiceMap, s::Selection)
    new_call = args[1]
    old_call = get_args(tr)[1]
    if key(old_call) != key(new_call)
        if (key(old_call) isa OUPMType && key(new_call) isa OUPMType) &&
                (typeof(key(old_call)) == typeof(key(new_call)))
            diff = MGFCallKeyChangeDiff(OUPMTypeIdxChange())
        else
            diff = MGFCallKeyChangeDiff(UnknownChange())
        end
    else
        diff = WorldUpdateDiff(new_call.world)[addr(new_call)][key(new_call)]
        # for these 3 diff types, just keep the diff type as is
        if diff !== NoChange() && diff !== ToBeUpdatedDiff() && diff !== WorldDiffedNoKeyChange()
            diff = MGFCallValChangeDiff(diff)
        end
    end
    return Gen.update(tr, args, (diff,), EmptyChoiceMap(), s)
end
@inline function Gen.update(tr::AutomaticIdxToIdLookupOrGenerateTrace, args::Tuple{MemoizedGenerativeFunctionCall}, argdiffs::Tuple{UnknownChange}, ::EmptyChoiceMap, s::Selection)
    new_call = args[1]
    old_call = get_args(tr)[1]
    if key(old_call) != key(new_call)
        # TODO: refactor; this part is the same as the beginning of the method defined just above this
        if (key(old_call) isa OUPMType && key(new_call) isa OUPMType) &&
                (typeof(key(old_call)) == typeof(key(new_call)))
            diff = MGFCallKeyChangeDiff(OUPMTypeIdxChange())
        else
            diff = MGFCallKeyChangeDiff(UnknownChange())
        end
    else
        diff = WorldDiffedNoKeyChange()
    end
    return Gen.update(tr, args, (diff,), EmptyChoiceMap(), s)
end