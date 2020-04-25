####################################
# LookupOrGenerate Parameter Types #
####################################

# TODO: should I include more information in the type for these?
struct MemoizedGenerativeFunction{WorldType, addr}
    world::WorldType
end
MemoizedGenerativeFunction(world::WorldType, addr::Symbol) where {WorldType} = MemoizedGenerativeFunction{WorldType, addr}()

struct MemoizedGenerativeFunctionCall{WorldType, addr}
    world::WorldType
    key
end
MemoizedGenerativeFunctionCall(world::WorldType, addr::Symbol, key) = MemoizedGenerativeFunctionCall{WorldType, addr}(world, key)

# world[:addr] gives a memoized gen function
# world[:addr][key] gives a memoized gen function call
Base.getproperty(world::World, addr::Symbol) = MemoizedGenerativeFunction(world, addr)
Base.getproperty(mgf::MemoizedGenerativeFunction, key) = MemoizedGenerativeFunctionCall(mgf.world, mgf.addr, key)

####################
# LookupOrGenerate #
####################

# TODO: could add some more information about the return type to the trace

struct LookupOrGenerateTrace <: Gen.Trace
    call::MemoizedGenerativeFunctionCall
    val
end

Gen.get_args(tr::LookupOrGenerateTrace) = (tr.world,)
Gen.get_retval(tr::LookupOrGenerateTrace) = tr.val
Gen.get_score(tr::LookupOrGenerateTrace) = 0.
Gen.get_gen_fn(tr::LookupOrGenerateTrace) = lookup_or_generate
Gen.project(tr::LookupOrGenerateTrace, selection::EmptySelection) = 0.
function Gen.get_choices(tr::LookupOrGenerateTrace)
    # TODO: static choicemap for performance?
    choicemap(
        (:val, tr.val),
        (metadata_addr(tr.call.world) => tr.call.addr, tr.call.key)
    )
end

struct LookupOrGenerate <: GenerativeFunction{Any, LookupOrGenerateTrace} end
const lookup_or_generate = LookupOrGenerate()

@inline (gen_fn::LookupOrGenerate)(args...) = get_retval(simulate(gen_fn, args))
@inline Gen.simulate(gen_fn::LookupOrGenerate, args::Tuple) = Gen.generate(gen_fn, args)[1]

function Gen.generate(gen_fn::LookupOrGenerate, args::Tuple{MemoizedGenerativeFunctionCall{WorldType, addr}}, constraints::EmptyChoiceMap) where {WorldType, addr}
    mgf_call, = args
    val = lookup_or_generate!(mgf_call.world, Call(addr, mgf_call.key))
    tr = LookupOrGenerateTrace(mgf_call, val)
    (tr, 0.)
end

function Gen.generate(gen_fn::Lookup, args::Tuple, constraints::ChoiceMap)
    error("generate(lookup_or_generate, ...) should only be called with empty constraints")
end

function Gen.update(tr::LookupOrGenerateTrace, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap)
    error("lookup_or_generate may not be updated with constraints.")
end

# TODO: update, regenerate

# TODO: gradients