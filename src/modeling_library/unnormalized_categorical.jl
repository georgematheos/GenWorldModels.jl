"""
    get_obj_sampler(obj_to_weight)
Returns a 0-ary function which samples an object
according to the unnormalized weights, and returns the pair
`(object, logprob)` where `logprob` is the log probability of
having sampled this object.
Calling is constructor function is `O(N)` in the number of objects.
"""
function get_obj_sampler(obj_to_weight)
    pairvec = collect(obj_to_weight)
    objvec = map(((x, _),) -> x, pairvec)
    weightvec = map(((_, y),) -> y, pairvec)
    total_weight = sum(weightvec)
    weightvec /= total_weight

    function sample_and_get_logprob()
        i = categorical(weightvec)
        obj = objvec[i]
        logprob = log(weightvec[i])
        return (obj, logprob)
    end
    return sample_and_get_logprob
end

# TODO: use some type of vector choicemap
struct UnnormalizedCategoricalChoiceMap{ObjType} <: Gen.AddressTree{Value}
    samples::PersistentVector{ObjType}
end
function Gen.get_subtree(c::UnnormalizedCategoricalChoiceMap, i::Integer)
    if i >= 1 && i <= length(c.samples)
        Value(c.samples[i])
    else
        EmptyAddressTree()
    end
end
Gen.get_subtree(c::UnnormalizedCategoricalChoiceMap, _) = EmptyAddressTree()
function Gen.get_subtrees_shallow(c::UnnormalizedCategoricalChoiceMap)
    ((i, Value(v)) for (i, v) in enumerate(c.samples))
end

struct UnnormalizedCategoricalTrace{ObjType} <: Gen.Trace
    args::Tuple{World, Int, AbstractDict{ObjType, <:Any}}
    samples::PersistentVector{ObjType}
    total_weight::Float64
    obj_to_indices::SetDict
    score::Float64
end
function UnnormalizedCategoricalTrace(
    args::Tuple{World, Int, AbstractDict{ObjType, <:Any}},
    samples::PersistentVector{ObjType},
    total_weight::Real,
    obj_to_indices::SetDict,
    score::Float64 
) where {ObjType}
    UnnormalizedCategoricalTrace{ObjType}(args, samples, Float64(total_weight), obj_to_indices, score)
end
Gen.get_gen_fn(::UnnormalizedCategoricalTrace) = unnormalized_categorical
Gen.get_args(tr::UnnormalizedCategoricalTrace) = tr.args
Gen.get_retval(tr::UnnormalizedCategoricalTrace) = tr.samples
Gen.get_score(tr::UnnormalizedCategoricalTrace) = tr.score
function Gen.get_choices(tr::UnnormalizedCategoricalTrace)
    v = values_to_concrete(tr.args[1], UnnormalizedCategoricalChoiceMap(tr.samples))
    try collect(get_subtrees_shallow(v))
    catch e
        display(tr.samples)
        display(tr.args[1].id_table)
        error()
    end
    return v
end
Gen.project(::UnnormalizedCategoricalTrace, ::EmptyAddressTree) = 0.

Base.getindex(tr::UnnormalizedCategoricalTrace, addr) =
    addr == :obj_to_indices ? tr.obj_to_indices :
    get_choices(tr)[addr]

struct UnnormalizedCategorical <: Gen.GenerativeFunction{
    PersistentVector,
    UnnormalizedCategoricalTrace
} end

"""
    list ~ unnormalized_categorical(world, num_samples, obj_to_weight)
Returns a vector with `num_samples` elements sampled with replacement from
`keys(obj_to_weight)`, where the probability that any index `i` contains
object `o` is `obj_to_weight[o] / sum(values(obj_to_weight))`.
"""
unnormalized_categorical = UnnormalizedCategorical()

function Gen.generate(
    ::UnnormalizedCategorical,
    args::Tuple{World, Integer, AbstractDict{ObjType, <:Any}},
    constraints::ChoiceMap
) where {ObjType}
    (world, num_samples, obj_to_weight) = args
    constraints = values_to_abstract(world, constraints)
    logprob_of_sampled = 0.
    total_logprob = 0.

    obj_sampler = get_obj_sampler(obj_to_weight)
    total_weight = sum(values(obj_to_weight))
    samples = PersistentVector{ObjType}()
    for i=1:num_samples
        constraint = get_subtree(constraints, i)
        if isempty(constraint)
            (obj, logprob) = obj_sampler()
            samples = push(samples, obj)
            logprob_of_sampled += logprob
            total_logprob += logprob
        else
            @assert has_value(constraint) "Constraint at index $i was a tree $constraint, not a value."
            obj = get_value(constraint)
            samples = push(samples, obj)
            total_logprob += log(obj_to_weight[obj]/total_weight)
        end
    end

    tr = UnnormalizedCategoricalTrace(args, samples, total_weight, item_to_indices(ObjType, samples), total_logprob)
    return (tr, total_logprob - logprob_of_sampled)
end

function Gen.update(
    tr::UnnormalizedCategoricalTrace,
    args::Tuple,
    (_, num_sample_diff, obj_to_weight_diff)::Tuple{Union{NoChange, <:GenWorldModels.WorldUpdateDiff}, Diff, Union{NoChange, <:DictDiff}},
    updatespec::UpdateSpec,
    eca::Selection
)
    (world, num_samples, obj_to_weight) = args
    updatespec = values_to_abstract(world, updatespec)

    num_changed = num_sample_diff !== NoChange()
    if obj_to_weight_diff === NoChange()
        obj_to_weight_diff = DictDiff(Dict(), Set(), Dict{Any, Diff}())
    end

    # we need to handle:
    # - changes in scores of unmodified samples due to the DictDiff
    # - added/removed samples due to a change in the number of samples
    # - changes to what we have sampled due to the updatespec
    
    # figure out the new total weight
    total_weight = tr.total_weight
    for (obj, weight) in obj_to_weight_diff.added
        total_weight += weight
    end
    for obj in obj_to_weight_diff.deleted
        total_weight -= get_args(tr)[3][obj]
    end
    for (obj, diff) in obj_to_weight_diff.updated
        if diff !== NoChange()
            total_weight -= get_args(tr)[3][obj]
            total_weight += obj_to_weight[obj]
        end
    end

    # update the samples
    objsampler = nothing
    updated = Dict{Int, Diff}()
    discard = choicemap()
    samples = tr.samples
    obj_to_indices = tr.obj_to_indices
    total_logprob = get_score(tr)
    log_q_ratio = 0. # log(q(tr -> new_tr unconstrained choices) / q(new_tr -> tr unconstrained choices))
    num_handled_for_obj = Dict() # TODO: accumulate in this dictionary as we go!
    total_num_handled = 0 # TODO: accumulate this
    for (i, subtree) in get_subtrees_shallow(updatespec)
        if i > num_samples
            error("Constraint provided for sample $i, but num_samples is $num_samples")
        elseif i > length(tr.samples)
            continue
        end
        if subtree isa Value
            obj = get_value(subtree)
            prob = obj_to_weight[obj]/total_weight
            total_logprob += log(prob)
        elseif subtree === AllSelection()
            if objsampler === nothing
                objsampler = get_obj_sampler(obj_to_weight)
            end
            (obj, logprob) = objsampler()
            log_q_ratio += logprob
            total_logprob += logprob
        else
            error("Unrecognized UpdateSpec at index $i: $subtree")
        end
        updated[i] = UnknownChange()
        discard[i] = tr.samples[i]
        total_logprob -= log(get_args(tr)[3][tr.samples[i]] / tr.total_weight)
        obj_to_indices = dissoc(obj_to_indices, tr.samples[i], i)
        obj_to_indices = assoc(obj_to_indices, obj, i)
        samples = assoc(samples, i, obj)
        num_handled_for_obj[obj] = get(num_handled_for_obj, obj, 0) + 1
        total_num_handled += 1
        if isempty(get_subtree(eca, i))
            log_q_ratio -= log(get_args(tr)[3][tr.samples[i]] / tr.total_weight)
        end
    end

    # handle additions/deletions
    if num_changed
        for i=get_args(tr)[2]:-1:(num_samples+1)
            # delete these samples
            obj = samples[i]
            samples = pop(samples)
            total_logprob -= log(get_args(tr)[3][tr.samples[i]] / tr.total_weight)
            obj_to_indices = dissoc(obj_to_indices, tr.samples[i], i)
            discard[i] = tr.samples[i]
        end
        for i=(get_args(tr)[2]+1):num_samples
            # generate these samples
            spec = get_subtree(updatespec, i)
            if spec isa Value
                obj = get_value(spec)
                prob = obj_to_weight[obj]/total_weight
                total_logprob += log(prob)
            elseif isempty(spec) || spec === AllSelection()
                if objsampler === nothing
                    objsampler = get_obj_sampler(obj_to_weight)
                end
                (obj, logprob) = objsampler()
                log_q_ratio += logprob
                total_logprob += logprob    
            else
                error("Unrecognized UpdateSpec at index $i: $spec")
            end
            samples = push(samples, obj)
            obj_to_indices = assoc(obj_to_indices, obj, i)
            num_handled_for_obj[obj] = get(num_handled_for_obj, obj, 0) + 1
            total_num_handled += 1
        end
    end

    # handle weight changes
    for (obj, diff) in obj_to_weight_diff.updated
        if diff !== NoChange()
            log_weight_change = log(obj_to_weight[obj]) - log(get_args(tr)[3][obj])
            num_unhandled_occurances = length(obj_to_indices[obj]) - get(num_handled_for_obj, obj, 0)
            total_logprob += num_unhandled_occurances * log_weight_change
        end
    end
    log_total_ratio = log(total_weight) - log(tr.total_weight)
    total_logprob += (num_samples - total_num_handled) * (-log_total_ratio)

    new_tr = UnnormalizedCategoricalTrace(args, samples, total_weight, obj_to_indices, total_logprob)
    diff = VectorDiff(num_samples, length(tr.samples), updated)
    weight = total_logprob - get_score(tr) - log_q_ratio

    old_world = get_args(tr)[1]
    return (new_tr, weight, diff, values_to_concrete(old_world, discard))
end

# TODO: other update signatures!