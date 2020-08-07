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

function Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap, ::Selection)
    error("lookup_or_generate may not be updated with constraints.")
end
function Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple, ::EmptyChoiceMap, ::Selection)
    error("Unrecognized update signature for lookup_or_generate.  Perhaps argdiff $(argdiffs[1]) is not recognized?")
end

# handle "regenerate"ing the trace
Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple, ::Selection, ::Selection) = update(tr, args, argdiffs, EmptyChoiceMap(), EmptySelection())

# no change
Gen.update(tr::LookupOrGenerateTrace, ::Tuple, ::Tuple{NoChange}, ::EmptyChoiceMap, ::Selection) = (tr, 0., NoChange(), EmptyChoiceMap())

function _update_via_new_generation(tr, args)
    new_tr = simulate(lookup_or_generate, args)
    retdiff = get_retval(new_tr) == get_retval(tr) ? NoChange() : UnknownChange()
    (new_tr, get_score(new_tr), retdiff, get_choices(tr))
end

SpecialMGFCallDiff = Union{ValueChangedDiff, ToBeUpdatedDiff, KeyChangedDiff}

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
    val = lookup_or_generate!(mgf_call.world, Call(addr(mgf_call), mgf_call.key), true)
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

# for a simple trace, if the key changes we effectively have to redo the whole lookup, so just generate a new trace
@inline function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, ::Tuple{KeyChangedDiff}, ::EmptyChoiceMap, ::Selection)
    _update_via_new_generation(tr, args)
end

@inline function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, ::Tuple{ToBeUpdatedDiff}, ::EmptyChoiceMap, ::Selection)
    mgf_call = args[1]
    # run a full update/generate cycle in the world for this call
    new_val = lookup_or_generate!(mgf_call.world, Call(addr(mgf_call), key(mgf_call)), false)
    new_tr = SimpleLookupOrGenerateTrace(mgf_call, new_val)
    retdiff = new_val == get_retval(tr) ? NoChange() : UnknownChange()
    (new_tr, 0., retdiff, EmptyChoiceMap())
end

# value has changed, but we don't need to update, and haven't changed key
function _simple_valchange_update(tr, args, argdiffs)
    valdiff = argdiffs[1].diff
    new_call = args[1]
    new_val = get_val(new_call.world, call(new_call))
    new_tr = SimpleLookupOrGenerateTrace(new_call, new_val)
    (new_tr, 0., valdiff, EmptyChoiceMap())
end
function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple, argdiffs::Tuple{ValueChangedDiff}, ::EmptyChoiceMap, ::Selection)
    _simple_valchange_update(tr, args, argdiffs)
end

# if we are looking up an origin or index, there is a chance we are making a lookup for an abstract object which has been
# deleted, and this call will be deleted.  In either case, we may need to return an invalid trace
OriginOrIndexMGFCall = Union{MemoizedGenerativeFunctionCall{<:Any, _get_index_addr}, MemoizedGenerativeFunctionCall{<:Any, _get_origin_addr}}
function Gen.update(tr::SimpleLookupOrGenerateTrace, args::Tuple{<:OriginOrIndexMGFCall}, argdiffs::Tuple{ValueChangedDiff}, ::EmptyChoiceMap, ::Selection)
    new_call = args[1]
    if has_val(new_call.world, call(new_call))
        _simple_valchange_update(tr, args, argdiffs)
    else
        (InvalidLookupOrGenerateTrace(new_call), -Inf, UnknownChange(), get_choices(tr))
    end
end

############################################################
# Static Gen functions underlying special-case trace types #
############################################################

# return a copy of `og_list` where the given `indices` have their values
# replaced by the values in `subs`, in order
function substitute_indices(og_list, indices, subs)
    lst = collect(OUPMObject, og_list)
    j=1
    for i in indices
        lst[i] = subs[j]
        j+=1
    end
    Tuple(lst)
end
function substitute_indices(og_list::Diffed{<:Any, NoChange}, indices::Diffed{<:Any, NoChange}, subs::Diffed{<:Any, NoChange})
    v = substitute_indices(strip_diff(og_list), strip_diff(indices), strip_diff(subs))
    Diffed(v, NoChange())
end
function substitute_indices(og_list::Diffed, indices::Diffed, subs::Diffed)
    v = substitute_indices(strip_diff(og_list), strip_diff(indices), strip_diff(subs))
    Diffed(v, UnknownChange())
end


Gen.@diffed_unary_function findall

@gen (static, diffs) function convert_key_to_abstract(mgf_call)
    obj = key(mgf_call)
    wrld = world(mgf_call)

    # convert origin to abstract
    concrete_indices = findall(map(x -> x isa ConcreteIndexOUPMObject, obj.origin))
    concrete_objs = map(i -> obj.origin[i], concrete_indices)
    newly_abstract_objs ~ Map(lookup_or_generate)(mgfcall_map(wrld[:abstract], concrete_objs))
    abstract_origin = substitute_indices(obj.origin, concrete_indices, newly_abstract_objs)

    to_lookup_obj::ConcreteIndexAbstractOriginOUPMObject = concrete_index_oupm_object(oupm_type_name(obj), abstract_origin, obj.idx)
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

Tuple(v::Diffed{<:Any, NoChange}) = Diffed(Tuple(strip_diff(v)), NoChange())
Tuple(v::Diffed) = Diffed(Tuple(strip_diff(v)), UnknownChange())

@gen (static, diffs) function tuple_idx_to_id_lookup_or_generate(mgf_call)
    tup = key(mgf_call)
    wrld = world(mgf_call)
    abstract_lst ~ Map(lookup_or_generate)(mgfcall_map(wrld[:abstract], tup))
    abstract_tup = Tuple(abstract_lst)
    val ~ lookup_or_generate(wrld[addr(mgf_call)][abstract_tup])
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
    argdiffs::Tuple{<:SpecialMGFCallDiff}, spec::EmptyAddressTree, ext_const_addrs::Selection
)
    new_tr, weight, retdiff, discard = update(tr.tr, args, argdiffs, spec, ext_const_addrs)
    (ToAbstractLookupOrGenerateTrace(new_tr), weight, retdiff, discard)
end

# if the update is changing the args so the concrete object now has fully abstract origin, we need to switch to a simple LoG trace
@inline function Gen.update(tr::ToAbstractLookupOrGenerateTrace, args::Tuple{ConcToAbsWithAbstractOriginCall}, ::Tuple{KeyChangedDiff}, ::EmptyAddressTree, ::Selection)
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
    argdiffs::Tuple{<:SpecialMGFCallDiff}, spec::EmptyAddressTree, ext_const_addrs::Gen.Selection
)
    new_tr, weight, retdiff, discard = update(tr.tr, args, argdiffs, spec, ext_const_addrs)
    (AutoKeyConversionLookupOrGenerateTrace(new_tr), weight, retdiff, discard)
end

# if we switch from a concrete key to a different lookup, or vice versa, we need to generate a new trace of a new type
@inline function Gen.update(tr::AutoKeyConversionLookupOrGenerateTrace, args::Tuple{MemoizedGenerativeFunctionCall}, ::Tuple{KeyChangedDiff}, ::EmptyAddressTree, ::Gen.Selection)
    _update_via_new_generation(tr, args)
end
@inline function Gen.update(tr::AutoKeyConversionLookupOrGenerateTrace, args::Tuple{ConcreteKeyMGFCall}, ::Tuple{KeyChangedDiff}, ::EmptyAddressTree, ::Gen.Selection)
    _update_via_new_generation(tr, args)
end

######################################
# Tuple argument auto key conversion #
######################################

"""
    AutoTupleKeyConversionLookupOrGenerateTrace
"""
struct AutoTupleKeyConversionLookupOrGenerateTrace <: LookupOrGenerateTrace
    tr::Gen.get_trace_type(tuple_idx_to_id_lookup_or_generate)
end
Gen.get_retval(tr::AutoTupleKeyConversionLookupOrGenerateTrace) = get_retval(tr.tr)
Gen.get_choices(tr::AutoTupleKeyConversionLookupOrGenerateTrace) = get_choices(tr.tr)
Gen.get_args(tr::AutoTupleKeyConversionLookupOrGenerateTrace) = get_args(tr.tr)

ConcreteIndexOUPMObjTupleKeyMGFCall = MemoizedGenerativeFunctionCall{<:Any, <:Any, <:Tuple{Vararg{<:ConcreteIndexOUPMObject}}}
EmptyTupleKeyMGFCall = MemoizedGenerativeFunctionCall{<:Any, <:Any, Tuple{}}

@inline function Gen.generate(::LookupOrGenerate, args::Tuple{ConcreteIndexOUPMObjTupleKeyMGFCall}, ::EmptyChoiceMap)
    tr, weight = generate(tuple_idx_to_id_lookup_or_generate, args, EmptyChoiceMap())
    (AutoTupleKeyConversionLookupOrGenerateTrace(tr), weight)
end
Gen.generate(::LookupOrGenerate, args::Tuple{EmptyTupleKeyMGFCall}, ::EmptyChoiceMap) = _simple_LoG_generate(args)
@inline function Gen.propose(::LookupOrGenerate, args::Tuple{ConcreteIndexOUPMObjTupleKeyMGFCall})
    propose(tuple_idx_to_id_lookup_or_generate, args)
end
Gen.propose(::LookupOrGenerate, args::Tuple{EmptyTupleKeyMGFCall}) = _simple_LoG_propose(args)
@inline function Gen.assess(::LookupOrGenerate, args::Tuple{ConcreteIndexOUPMObjTupleKeyMGFCall}, ::EmptyChoiceMap)
    assess(tuple_idx_to_id_lookup_or_generate, args, EmptyChoiceMap())
end
Gen.assess(::LookupOrGenerate, args::Tuple{EmptyTupleKeyMGFCall}, ::EmptyChoiceMap) = _simple_LoG_assess(args)
@inline function Gen.update(
    tr::AutoTupleKeyConversionLookupOrGenerateTrace,
    args::Tuple{ConcreteIndexOUPMObjTupleKeyMGFCall},
    argdiffs::Tuple{<:SpecialMGFCallDiff}, spec::EmptyAddressTree, ext_const_addrs::Gen.Selection
)
    new_tr, weight, retdiff, discard = update(tr.tr, args, argdiffs, spec, ext_const_addrs)
    (AutoTupleKeyConversionLookupOrGenerateTrace(new_tr), weight, retdiff, discard)
end

# if we switch from a concrete key to a different lookup, or vice versa, we need to generate a new trace of a new type
@inline function Gen.update(tr::AutoTupleKeyConversionLookupOrGenerateTrace, args::Tuple{MemoizedGenerativeFunctionCall}, ::Tuple{KeyChangedDiff}, ::EmptyAddressTree, ::Gen.Selection)
    _update_via_new_generation(tr, args)
end
@inline function Gen.update(tr::AutoTupleKeyConversionLookupOrGenerateTrace, args::Tuple{ConcreteIndexOUPMObjTupleKeyMGFCall}, ::Tuple{KeyChangedDiff}, ::EmptyAddressTree, ::Gen.Selection)
    _update_via_new_generation(tr, args)
end

######################################
# UnknownChange `update` dispatching #
######################################

# If this `lookup_or_generate` call is made from a generative function which doesn't propagate diffs, we'll get an `UnknownChange`.
# In this case, we should manually propagate the diff information so we can dispatch to the correct update behavior.
function Gen.update(tr::LookupOrGenerateTrace, args::Tuple{<:MemoizedGenerativeFunctionCall}, ::Tuple{UnknownChange}, ::EmptyAddressTree, s::Selection)
    old_call = get_args(tr)[1]

    old_key = key(old_call)
    new_call = args[1]
    new_key = key(new_call)
    keydiff = old_key == new_key ? NoChange() : UnknownChange()
    diffed_key = Diffed(new_key, keydiff)

    wrld = world(new_call)
    @assert wrld.state isa UpdateWorldState "World should be in an `UpdateWorldState` when an update to `lookup_or_generate` occurs!"
    diffed_world = Diffed(wrld, WorldUpdateDiff())
    diffed_mgf_call = diffed_world[addr(new_call)][diffed_key]
    return Gen.update(tr, (strip_diff(diffed_mgf_call),), (get_diff(diffed_mgf_call),), EmptyAddressTree(), s)
end