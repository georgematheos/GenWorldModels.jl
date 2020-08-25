@gen (static) function _get_single_origin_object_set(typename, world, origin, num)
    conc_objs = map(x -> ConcreteIndexOUPMObject{typename}(origin, x), 1:num)
    objs ~ Map(lookup_or_generate)(mgfcall_map(world[_get_abstract_addr], conc_objs))
    return SimpleObjectSet(PersistentSet(objs))
end
@load_generated_functions()

struct GetSingleOriginObjectSetTrace <: Gen.Trace
    gen_fn
    tr::Gen.get_trace_type(_get_single_origin_object_set)
end
Gen.get_score(tr::GetSingleOriginObjectSetTrace) = get_score(tr.tr)
Gen.get_gen_fn(tr::GetSingleOriginObjectSetTrace) = tr.gen_fn
Gen.get_retval(tr::GetSingleOriginObjectSetTrace) = get_retval(tr.tr)
Gen.get_args(tr::GetSingleOriginObjectSetTrace) = get_args(tr.tr)[2:end]
Gen.get_choices(tr::GetSingleOriginObjectSetTrace) = get_choices(tr.tr)
Gen.project(::GetSingleOriginObjectSetTrace, ::Selection) = 0.

# GetOriginlessObjectSet(typename)(world, origin, num)
struct GetSingleOriginObjectSet <: Gen.GenerativeFunction{SimpleObjectSet, GetSingleOriginObjectSetTrace}
    typename::Symbol
end

function Gen.simulate(gen_fn::GetSingleOriginObjectSet, (world, origin, num)::Tuple)
    tr = simulate(_get_single_origin_object_set, (gen_fn.typename, world, origin, num))
    GetSingleOriginObjectSetTrace(gen_fn, tr)
end
Gen.generate(gen_fn::GetSingleOriginObjectSet, args::Tuple, ::EmptyChoiceMap) = (simulate(gen_fn, args), 0.)

function Gen.update(tr::GetSingleOriginObjectSetTrace, args::Tuple, argdiffs::Tuple, spec::UpdateSpec, ext_const_addrs::Selection)
    gen_fn = Gen.get_gen_fn(tr)
    new_tr, weight, retdiff, discard = update(tr.tr, (gen_fn.typename, args...), (NoChange(), argdiffs...), spec, ext_const_addrs)

    if retdiff === UnknownChange()
        new_set = get_retval(new_tr)
        old_set = get_retval(tr)
        added = Set((x for x in new_set if !(x in old_set)))
        removed = Set((x for x in old_set if !(x in new_set)))
        retdiff = SetDiff(added, removed)
    end

    (GetSingleOriginObjectSetTrace(gen_fn, new_tr), weight, retdiff, discard)
end

### We also provide a wrapper to this where we pass the `typename` in as an argument
### to the generative function.
### TODO: is there a way we can avoid needing this?

struct GetSingleOriginObjectSetTypenameArgTrace <: Gen.Trace
    typename
    tr::GetSingleOriginObjectSetTrace
end
Gen.get_score(tr::GetSingleOriginObjectSetTypenameArgTrace) = get_score(tr.tr)
Gen.get_retval(tr::GetSingleOriginObjectSetTypenameArgTrace) = get_retval(tr.tr)
Gen.get_args(tr::GetSingleOriginObjectSetTypenameArgTrace) = (tr.typename, get_args(tr.tr))
Gen.get_choices(tr::GetSingleOriginObjectSetTypenameArgTrace) = get_choices(tr.tr)
Gen.project(::GetSingleOriginObjectSetTypenameArgTrace, ::Selection) = 0.

struct GetSingleOriginObjectSetTypenameArg <: Gen.GenerativeFunction{SimpleObjectSet, GetSingleOriginObjectSetTypenameArgTrace} end
get_single_origin_object_set = GetSingleOriginObjectSetTypenameArg()
Gen.get_gen_fn(::GetSingleOriginObjectSetTypenameArgTrace) = get_single_origin_object_set
function Gen.simulate(gen_fn::GetSingleOriginObjectSetTypenameArg, args::Tuple)
    (typename, other_args) = (args[1], args[2:end])
    tr = simulate(GetSingleOriginObjectSet(typename), other_args)
    return GetSingleOriginObjectSetTypenameArgTrace(typename, tr)
end
Gen.generate(gen_fn::GetSingleOriginObjectSetTypenameArg, args::Tuple, ::EmptyChoiceMap) = (simulate(gen_fn, args), 0.)
function Gen.update(tr::GetSingleOriginObjectSetTypenameArgTrace, args::Tuple, argdiffs::Tuple, spec::UpdateSpec, eca::Selection)
    typename, other_args = args[1], args[2:end]
    other_diffs = argdiffs[2:end]
    @assert typename === tr.typename "Cannot change typename for object set!"
    (new_tr, weight, retdiff, discard) = update(tr.tr, other_args, other_diffs, spec, eca)
    new_tr = GetSingleOriginObjectSetTypenameArgTrace(typename, new_tr)
    return (new_tr, weight, retdiff, discard)
end

