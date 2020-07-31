####################
# LookupOrGenerate #
####################

# TODO: could add some more information about the return type to the trace
# TODO: once "generative methods" are implemented, it should be possible to use this to simplify this file
# and switching between different "methods" for `lookup_or_generate`

abstract type LookupOrGenerateTrace <: Gen.Trace end

#=
Depending on the call, there are a few ways we perform `lookup_or_generate`.
1. Simple `lookup_or_generate` - for a normal MGF call.
2. MGF call with a `ConcreteIndexOUPMObject` which `lookup_or_generate` converts to abstract form.
3. Abstract object lookup - must convert all the origin objects to abstract.
4. Invalid `lookup_or_generate` trace -- after a lookup is updated in such a way that the lookup must be removed
for the `UsingWorld` trace to remain valid.
=#

### Shared GFI Methods ###

# gfi methods with the same implementation for all valid concrete types of `LookupOrGenerateTrace`
Gen.get_score(tr::LookupOrGenerateTrace) = 0.
Gen.get_gen_fn(tr::LookupOrGenerateTrace) = lookup_or_generate
Gen.project(::LookupOrGenerateTrace, ::EmptySelection) = 0.
Gen.project(::LookupOrGenerateTrace, ::Selection) = 0.

"""
    InvalidLookupOrGenerateTrace
"""
struct InvalidLookupOrGenerateTrace <: LookupOrGenerateTrace
    call::MemoizedGenerativeFunctionCall
end
Gen.get_score(::InvalidLookupOrGenerateTrace) = -Inf
Gen.project(::InvalidLookupOrGenerateTrace, ::EmptySelection) = -Inf
Gen.project(::InvalidLookupOrGenerateTrace, ::Selection) = -Inf
Gen.get_args(tr::InvalidLookupOrGenerateTrace) = (tr.call,)
Gen.get_retval(::InvalidLookupOrGenerateTrace) = nothing
Gen.get_choices(::InvalidLookupOrGenerateTrace) = EmptyAddressTree()

"""
    lookup_or_generate(world[address][key])

Look up the value for the call with argument `key` to memoized generative function
with name `address` in the world, or generate a value for it if none currently exists.
"""
struct LookupOrGenerate <: GenerativeFunction{Any, LookupOrGenerateTrace} end
const lookup_or_generate = LookupOrGenerate()

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

function _update_via_new_generation(tr, args)
    new_tr = simulate(lookup_or_generate, args)
    retdiff = get_retval(new_tr) == get_retval(tr) ? NoChange() : UnknownChange()
    (new_tr, get_score(new_tr), retdiff, get_choices(tr))
end

###########################
# Simple LookupOrGenerate #
###########################

"""
    SimpleLookupOrGenerateTrace
"""
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

function _simple_LoG_generate(args)
    mgf_call, = args
    val = lookup_or_generate!(mgf_call.world, Call(addr(mgf_call), mgf_call.key); reason_for_call=:generate)
    tr = SimpleLookupOrGenerateTrace(mgf_call, val)
    (tr, 0.)
end

function _simple_LoG_propose(args)
    mgf_call, = args
    retval = lookup_or_propose!(mgf_call.world, Call(addr(mgf_call), mgf_call.key))
    
    # we can return an EmptyChoiceMap for choices since we don't need to track
    # dependencies for a propose value, since the generated world is discarded afterwards.
    (EmptyChoiceMap(), 0., retval)
end

function _simple_LoG_assess(args)
    mgf_call, = args
    retval = lookup_or_assess!(mgf_call.world, Call(addr(mgf_call), mgf_call.key))

    (0., retval)
end

Gen.generate(::LookupOrGenerate, args::Tuple{MemoizedGenerativeFunctionCall{<:Any, <:Any, <:Any}}, ::EmptyChoiceMap) = _simple_LoG_generate(args)
Gen.propose(::LookupOrGenerate, args::Tuple) = _simple_LoG_propose(args)
Gen.assess(::LookupOrGenerate, args::Tuple, ::EmptyChoiceMap) = _simple_LoG_assess(args)

### Simple LookupOrGenerate Update ###

function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap, ::Selection)
    error("lookup_or_generate may not be updated with constraints.")
end
function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, argdiffs::Tuple, ::EmptyChoiceMap, ::Selection)
    error("Unrecognized update signature for lookup_or_generate.  Perhaps argdiff $(argdiffs[1]) is not recognized?")
end

# handle "regenerate"ing the trace
Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, argdiffs::Tuple, ::Selection, ::Selection) = update(tr, args, argdiffs, EmptyChoiceMap(), EmptySelection())

# no change
Gen.update(tr::SimpleLookupOrGenerateTrace, ::Tuple, ::Tuple{NoChange}, ::EmptyChoiceMap, ::Selection) = (tr, 0., NoChange(), EmptyChoiceMap())

### SimpleLookupOrGenerateTrace updates ###

function _simple_keychange_update(tr::SimpleLookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{MGFCallKeyChangeDiff})
    mgf_call, = args

    # run a full update/generate cycle in the world for this call
    new_val = lookup_or_generate!(mgf_call.world, Call(addr(mgf_call), key(mgf_call)); reason_for_call=:key_change)
    
    # the key looked up is the only thing exposed in the choicemap; this has changed so the whole old choicemap is the discard
    discard = get_choices(tr)
    
    new_tr = SimpleLookupOrGenerateTrace(mgf_call, new_val)
    retdiff = new_val == get_retval(tr) ? NoChange() : UnknownChange()
    (new_tr, 0., retdiff, discard)
end

function _simple_to_be_updated_update(tr::SimpleLookupOrGenerateTrace, args::Tuple)
    mgf_call = args[1]
    # run a full update/generate cycle in the world for this call
    new_val = lookup_or_generate!(mgf_call.world, Call(addr(mgf_call), key(mgf_call)); reason_for_call=:to_be_updated)
    new_tr = SimpleLookupOrGenerateTrace(mgf_call, new_val)
    retdiff = new_val == get_retval(tr) ? NoChange() : UnknownChange()
    (new_tr, 0., retdiff, EmptyChoiceMap())
end

@inline function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{MGFCallKeyChangeDiff}, ::EmptyChoiceMap, ::Selection)
    _simple_keychange_update(tr, args, argdiffs)
end

@inline function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, ::Tuple{ToBeUpdatedDiff}, ::EmptyChoiceMap, ::Selection)
    _simple_to_be_updated_update(tr, args)
end

# ID lookup for OUPM type
# if the ID association has changed, we ultimately want to perform a `lookup_or_generate!` call in case we need to 
# generate a new ID--so we can use the logic for the `ToBeUpdatedDiff` for this.
# function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple{MemoizedGenerativeFunctionCall{<:Any, <:ConcreteIndexOUPMObject}}, argdiffs::Tuple{IDAssociationChanged}, ::EmptyChoiceMap, ::Selection)
#     update(tr, args, (ToBeUpdatedDiff(),), EmptyChoiceMap(), EmptySelection())
# end

### TODO: diffs below here ###

# INDEX lookup for OUPM type
# either the identifier has been moved, in which case we can simply treat this as a call whose val has changed,
# or the identifier has been deleted, in which case this call is invalidated, so we return 0 probability
# function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple{MemoizedGenerativeFunctionCall{<:Any, _get_index_addr}}, argdiffs::Tuple{IDAssociationChanged}, ::EmptyChoiceMap, ::Selection)
#     if has_val(args[1].world, call(args[1]))
#         update(tr, args, (MGFCallValChangeDiff(UnknownChange()),), EmptyChoiceMap(), EmptySelection())
#     else
#         (InvalidLookupOrGenerateTrace(args[1]), -Inf, UnknownChange(), get_choices(tr))
#     end
# end

# value has changed, but we don't need to update, and haven't changed key
function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{MGFCallValChangeDiff}, ::EmptyChoiceMap, ::Selection)
    valdiff = argdiffs[1].diff
    new_call = args[1]
    new_val = get_val(new_call.world, call(new_call))
    new_tr = SimpleLookupOrGenerateTrace(new_call, new_val)
    (new_tr, 0., valdiff, EmptyChoiceMap())
end

############################################################
# Static Gen functions underlying special-case trace types #
############################################################

function substitute_indices(og_list, indices, subs)
    lst = collect(OUPMObject, og_list)
    j=1
    for i in indices
        lst[i] = subs[j]
        j+=1
    end
    Tuple(lst)
end

@gen (static, diffs) function convert_key_to_abstract(mgf_call)
    obj = key(mgf_call)
    wrld = world(mgf_call)
    concrete_indices = findall(x -> x isa ConcreteIndexOUPMObject, obj.origin)
    newly_abstract_objs ~ Map(lookup_or_generate)([wrld[:abstract][obj.origin[i]] for i in concrete_indices])
    abstract_origin = substitute_indices(obj.origin, concrete_indices, newly_abstract_objs)
    to_lookup_obj::ConcreteIndexAbstractOriginOUPMObject = ConcreteIndexOUPMObject{typename(obj)}(abstract_origin, obj.idx)
    abstract_obj ~ lookup_or_generate(wrld[:abstract][to_lookup_obj])
    return abstract_obj
end

@gen (static, diffs) function idx_to_id_lookup_or_generate(mgf_call)
    concrete_obj = key(mgf_call)
    wrld = world(mgf_call)
    abstract_obj ~ lookup_or_generate(wrld[:abstract][concrete_obj])
    val ~ lookup_or_generate(wrld[addr(mgf_call)][abstract_obj])
    return val 
end

@load_generated_functions()

##################################################
# Explicit concrete --> absract LookupOrGenerate #
##################################################


"""
    ToAbstractLookupOrGenerateTrace
"""
struct ToAbstractLookupOrGenerateTrace <: LookupOrGenerateTrace
    tr::Gen.get_trace_type(convert_key_to_abstract)
end

Gen.get_retval(tr::ToAbstractLookupOrGenerateTrace) = get_retval(tr.tr)
Gen.get_choices(tr::ToAbstractLookupOrGenerateTrace) = get_choices(tr.tr)
Gen.get_args(tr::ToAbstractLookupOrGenerateTrace) = get_args(tr.tr)

ConcToAbsMGFCall = MemoizedGenerativeFunctionCall{<:Any, _get_abstract_addr, <:ConcreteIndexOUPMObject}
ConcToAbsWithAbstractOriginCall = MemoizedGenerativeFunctionCall{<:Any, _get_abstract_addr, <:ConcreteIndexAbstractOriginOUPMObject}

# if we are looking up the abstract form of an object with abstract origin, we don't need any special `lookup_or_generate`
# logic, so create a "simple" lookup or generate trace
@inline Gen.generate(::LookupOrGenerate, args::Tuple{ConcToAbsWithAbstractOriginCall}, ::EmptyChoiceMap) = _simple_LoG_generate(args)
@inline Gen.propose(::LookupOrGenerate, args::Tuple{ConcToAbsWithAbstractOriginCall}) = _simple_LoG_propose(args)
@inline Gen.assess(::LookupOrGenerate, args::Tuple{ConcToAbsWithAbstractOriginCall}, ::EmptyChoiceMap) = _simple_LoG_assess(args)

# if we update a simple trace Conc-->Abs where we now have a concrete origin element, we need to swap traces to handle this
@inline function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple{ConcToAbsMGFCall}, ::Tuple{MGFCallKeyChangeDiff}, ::EmptyChoiceMap, ::Selection)
    _update_via_new_generation(tr, args)
end
# if we update a simple trace Conc-->abs and we still have a fully abstract origin, we can do the normal simple update
@inline function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple{ConcToAbsWithAbstractOriginCall}, argdiffs::Tuple{MGFCallKeyChangeDiff}, ::EmptyChoiceMap, ::Selection)
    _simple_keychange_update(tr, args, argdiffs)
end

@inline function Gen.generate(::LookupOrGenerate, args::Tuple{ConcToAbsMGFCall}, ::EmptyChoiceMap)
    tr, weight = generate(convert_key_to_abstract, args, EmptyChoiceMap())
    (ToAbstractLookupOrGenerateTrace(tr), weight)
end
@inline function Gen.propose(::LookupOrGenerate, args::Tuple{ConcToAbsMGFCall})
    propose(convert_key_to_abstract, args)
end
@inline function Gen.assess(::LookupOrGenerate, args::Tuple{ConcToAbsMGFCall}, ::EmptyChoiceMap)
    assess(convert_key_to_abstract, args, EmptyChoiceMap())
end
@inline function Gen.update(
    tr::ToAbstractLookupOrGenerateTrace,
    args::Tuple{ConcToAbsMGFCall},
    argdiffs::Tuple{<:Gen.Diff}, spec::Gen.UpdateSpec, ext_const_addrs::Gen.Selection
)
    new_tr, weight, retdiff, discard = update(tr.tr, args, argdiffs, spec, ext_const_addrs)
    (ToAbstractLookupOrGenerateTrace(new_tr), weight, retdiff, discard)
end

# if the update is changing the args so the concrete object now has fully abstract origin, we need to switch to a simple LoG trace
@inline function Gen.update(tr::ToAbstractLookupOrGenerateTrace, args::Tuple{ConcToAbsWithAbstractOriginCall}, ::Tuple{<:Gen.Diff}, ::Gen.UpdateSpec, ::Gen.Selection)
    _update_via_new_generation(tr, args)
end

###############################################################
# Automatic Concrete --> Abstract Conversion LookupOrGenerate #
###############################################################

"""
    AutoKeyConversionLookupOrGenerateTrace
"""
struct AutoKeyConversionLookupOrGenerateTrace <: LookupOrGenerateTrace
    tr::Gen.get_trace_type(idx_to_id_lookup_or_generate)
end
Gen.get_retval(tr::AutoKeyConversionLookupOrGenerateTrace) = get_retval(tr.tr)
Gen.get_choices(tr::AutoKeyConversionLookupOrGenerateTrace) = get_choices(tr.tr)
Gen.get_args(tr::AutoKeyConversionLookupOrGenerateTrace) = get_args(tr.tr)

ConcreteKeyMGFCall = MemoizedGenerativeFunctionCall{<:Any, <:Any, <:ConcreteIndexOUPMObject}

@inline function Gen.generate(::LookupOrGenerate, args::Tuple{ConcreteKeyMGFCall}, ::EmptyChoiceMap)
    tr, weight = generate(idx_to_id_lookup_or_generate, args, EmptyChoiceMap())
    (AutoKeyConversionLookupOrGenerateTrace(tr), weight)
end
@inline function Gen.propose(::LookupOrGenerate, args::Tuple{ConcreteKeyMGFCall})
    propose(idx_to_id_lookup_or_generate, args)
end
@inline function Gen.assess(::LookupOrGenerate, args::Tuple{ConcreteKeyMGFCall}, ::EmptyChoiceMap)
    assess(idx_to_id_lookup_or_generate, args, EmptyChoiceMap())
end
@inline function Gen.update(
    tr::AutoKeyConversionLookupOrGenerateTrace,
    args::Tuple{ConcreteKeyMGFCall},
    argdiffs::Tuple{<:Gen.Diff}, spec::EmptyAddressTree, ext_const_addrs::Gen.Selection
)
    new_tr, weight, retdiff, discard = update(tr.tr, args, argdiffs, spec, ext_const_addrs)
    (AutoKeyConversionLookupOrGenerateTrace(new_tr), weight, retdiff, discard)
end

# if we switch from a concrete key to a different lookup, or vice versa, we need to generate a new trace of a new type
@inline function Gen.update(tr::AutoKeyConversionLookupOrGenerateTrace, args::Tuple{MemoizedGenerativeFunctionCall}, ::Tuple{<:Gen.Diff}, ::Gen.UpdateSpec, ::Gen.Selection)
    _update_via_new_generation(tr, args)
end
@inline function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple{ConcreteKeyMGFCall}, ::Tuple{MGFCallKeyChangeDiff}, ::Gen.UpdateSpec, ::Gen.Selection)
    _update_via_new_generation(tr, args)
end

########################################
# UnknownChange `update` dispatching ###
########################################

# UnknownChange  is likely caused by use of a dynamic generative function which doesn't propagate diffs
# we need to figure out which of the other update functions we should dispatch to
# this function just replicates the diff propagation behavior implemented above in the Base.getindex methods
@inline function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple{MemoizedGenerativeFunctionCall}, argdiffs::Tuple{UnknownChange}, ::EmptyChoiceMap, s::Selection)
    new_call = args[1]
    old_call = get_args(tr)[1]
    if key(old_call) != key(new_call)
        if (key(old_call) isa OUPMObject && key(new_call) isa OUPMObject) &&
                (typeof(key(old_call)) == typeof(key(new_call)))
            diff = MGFCallKeyChangeDiff(OUPMTypeIdxChange())
        else
            diff = MGFCallKeyChangeDiff(UnknownChange())
        end
    else
        diff = WorldUpdateDiff(new_call.world)[addr(new_call)][key(new_call)]
        # for these 3 diff types, just keep the diff type as is
        if diff !== NoChange() && diff !== ToBeUpdatedDiff() && diff !== WorldDiffedNoKeyChange()# && diff !== IDAssociationChanged()
            diff = MGFCallValChangeDiff(diff)
        end
    end
    return Gen.update(tr, args, (diff,), EmptyChoiceMap(), s)
end

function _autokeyconvert_unknownchange_update(tr, args, s)
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

@inline function Gen.update(tr::AutoKeyConversionLookupOrGenerateTrace, args::Tuple{MemoizedGenerativeFunctionCall}, ::Tuple{UnknownChange}, ::EmptyChoiceMap, s::Selection)
    _autokeyconvert_unknownchange_update(tr, args, s)
end
@inline function Gen.update(tr::AutoKeyConversionLookupOrGenerateTrace, args::Tuple{ConcreteKeyMGFCall}, ::Tuple{UnknownChange}, ::EmptyChoiceMap, s::Selection)
    _autokeyconvert_unknownchange_update(tr, args, s)
end