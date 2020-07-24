"""
Code largely borrowed from Gen's "trace transform DSL" implementation.
"""

import ForwardDiff
import MacroTools
import LinearAlgebra
import Parameters: @with_kw

struct OUPMInvolutionDSLProgram
    fn!::Function
end

struct ModelInputTraceToken{T}
    args::T
end

struct AuxInputTraceToken{T}
    args::T
end

struct ModelInputTraceRetValToken
end

struct AuxInputTraceRetValToken
end

struct ModelOutputTraceToken 
end

struct AuxOutputTraceToken
end

struct ModelInputAddress{T}
    addr::T
end

struct AuxInputAddress{T}
    addr::T
end

struct ModelOutputAddress{T}
    addr::T
end

struct AuxOutputAddress{T}
    addr::T
end

Base.getindex(::ModelInputTraceToken, addr) = ModelInputAddress(addr) # model_in[addr]
Base.getindex(::ModelOutputTraceToken, addr) = ModelOutputAddress(addr) # model_out[addr]
Base.getindex(::AuxInputTraceToken, addr) = AuxInputAddress(addr) # aux_in[addr]
Base.getindex(::AuxOutputTraceToken, addr) = AuxOutputAddress(addr) # aux_out[addr]
Base.getindex(::ModelInputTraceToken) = ModelInputTraceRetvalToken() # model_in[]
Base.getindex(::AuxInputTraceToken) = AuxInputTraceRetValToken() # aux_in[]
Gen.get_args(token::ModelInputTraceToken) = token.args # get_args(model_in)
Gen.get_args(token::AuxInputTraceToken) = token.args # get_args(aux_in)

const bij_state = gensym("bij_state")

"""
    @oupm_involution f[(params...)] (old_model_tr , fwd_proposal_tr) to (new_model_tr, bwd_proposal_tr) begin
        ...
    end

Write a program in the [Trace Transform DSL](@ref).
"""
macro oupm_involution(f_expr, src_expr, to_symbol::Symbol, dest_expr, body)
    syntax_err = "valid syntactic forms:\n@oupm_involution f (old_model_tr , fwd_proposal_tr) to (new_model_tr, bwd_proposal_tr) begin .. end\n@transform f(..) (old_model_tr , fwd_proposal_tr) to (new_model_tr, bwd_proposal_tr) begin .. end"
    err = false
    if MacroTools.@capture(f_expr, f_(args__))
    elseif MacroTools.@capture(f_expr, f_)
        args = []
    else
        err = true
    end
    err = err || (to_symbol != :to)
    if !MacroTools.@capture(src_expr, (model_in_, aux_in_))
        err = true
    end
    if !MacroTools.@capture(dest_expr, (model_out_, aux_out_))
        err = true
    end

    fn! = gensym("$(esc(f))_fn!")

    return quote

        # mutates the state
        function $fn!(
                $(esc(bij_state))::Union{FirstPassState,JacobianPassState},
                $(map(esc, args)...))
            model_args = get_args($(esc(bij_state)).model_trace)
            aux_args = get_args($(esc(bij_state)).aux_trace)
            $(esc(model_in)) = ModelInputTraceToken(model_args)
            $(esc(model_out)) = ModelOutputTraceToken()
            $(esc(aux_in)) = AuxInputTraceToken(aux_args)
            $(esc(aux_out)) = AuxOutputTraceToken()
            $(esc(body))
            return nothing
        end

        Core.@__doc__ $(esc(f)) = OUPMInvolutionDSLProgram($fn!)

    end
end

macro tcall(ex)
    MacroTools.@capture(ex, f_(args__)) || error("expected syntax: f(..)")
    return quote $(esc(f)).fn!($(esc(bij_state)), $(map(esc, args)...)) end
end

# handlers

struct DiscreteAnn end
struct ContinuousAnn end

const DISCRETE = [:discrete, :disc]
const CONTINUOUS = [:continuous, :cont]

function typed(annotation::Symbol)
    if annotation in DISCRETE
        return DiscreteAnn()
    elseif annotation in CONTINUOUS
        return ContinuousAnn()
    else
        error("error")
    end
end

"""
    @read(<source>, <annotation>)

Macro for reading the value of a random choice from an input trace in the [Trace Transform DSL](@ref).

<source> is of the form <trace>[<addr>] where <trace> is an input trace, and <annotation> is either :discrete or :continuous.
"""
macro read(src, ann::QuoteNode)
    return quote read($(esc(bij_state)), $(esc(src)), $(esc(typed(ann.value)))) end
end

"""
    @write(<destination>, <value>, <annotation>)

Macro for writing the value of a random choice to an output trace in the [Trace Transform DSL](@ref).

<destination> is of the form <trace>[<addr>] where <trace> is an input trace, and <annotation> is either :discrete or :continuous.
"""
macro write(dest, val, ann::QuoteNode)
    return quote
        write($(esc(bij_state)), $(esc(dest)), $(esc(val)), $(esc(typed(ann.value))))
    end
end

"""
    @copy(<source>, <destination>)

Macro for copying the value of a random choice (or a whole namespace of random choices) from an input trace to an output trace in the [Trace Transform DSL](@ref).

<destination> is of the form <trace>[<addr>] where <trace> is an input trace, and <annotation> is either :discrete or :continuous.
"""
macro copy(src, dest)
    return quote copy($(esc(bij_state)), $(esc(src)), $(esc(dest))) end
end

macro birth(ObjectType, idx)
    quote
        move = BirthMove($(esc(ObjectType)), $(esc(idx)))
        apply_oupm_move($(esc(bij_state)), move)
    end
end
macro death(ObjectType, idx)
    quote
        move = DeathMove($(esc(ObjectType)), $(esc(idx)))
        apply_oupm_move($(esc(bij_state)), move)
    end
end
macro split(ObjectType, from_idx, to_idx1, to_idx2)
    quote
        move = SplitMove($(esc(ObjectType)), $(esc(from_idx)), $(esc(to_idx1)), $(esc(to_idx2)))
        apply_oupm_move($(esc(bij_state)), move)
    end
end
macro merge(ObjectType, to_idx, from_idx1, from_idx2)
    quote
        move = MergeMove($(esc(ObjectType)), $(esc(to_idx)), $(esc(from_idx1)), $(esc(from_idx2)))
        apply_oupm_move($(esc(bij_state)), move)
    end
end
macro move(ObjectType, from_idx, to_idx)
    quote
        move = MoveMove($(esc(ObjectType)), $(esc(from_idx)), $(esc(to_idx)))
        apply_oupm_move($(esc(bij_state)), move)
    end
end


# TODO make more consistent by allowing us to read any hierarchical address,
# including return values of intermediate calls, not just the top-level call.

# TODO add haskey(model_in, addr), and haskey(aux_in, addr)

################################
# first pass through transform #
################################

mutable struct FirstPassResults
    update_spec::UpdateWithOUPMMovesSpec

    "output proposal choice map ``u'``"
    u_back::ChoiceMap
    
    t_cont_reads::Dict
    u_cont_reads::Dict
    t_cont_writes::Dict
    u_cont_writes::Dict
    t_copy_reads::DynamicSelection
    u_copy_reads::DynamicSelection
end

function FirstPassResults()
    return FirstPassResults(
        UpdateWithOUPMMovesSpec((), choicemap()), choicemap(),
        Dict(), Dict(), Dict(), Dict(),
        DynamicSelection(), DynamicSelection())
end

struct FirstPassState

    "trace containing the input model choice map ``t``"
    model_trace

    "the input proposal choice map ``u``"
    aux_trace

    results::FirstPassResults
end

function FirstPassState(model_trace, aux_trace)
    return FirstPassState(model_trace, aux_trace, FirstPassResults())
end

function run_first_pass(transform::OUPMInvolutionDSLProgram, model_trace, aux_trace)
    state = FirstPassState(model_trace, aux_trace)
    transform.fn!(state) # TODO allow for other args to top-level transform function
    return state.results
end

function read(state::FirstPassState, src::ModelInputTraceRetValToken, ::DiscreteAnn)
    return get_retval(state.model_trace)
end

function read(state::FirstPassState, src::AuxInputTraceRetValToken, ::DiscreteAnn)
    return get_retval(state.aux_trace)
end

function read(state::FirstPassState, src::ModelInputAddress, ::DiscreteAnn)
    addr = src.addr
    return state.model_trace[addr]
end

function read(state::FirstPassState, src::ModelInputAddress, ::ContinuousAnn)
    addr = src.addr
    state.results.t_cont_reads[addr] = state.model_trace[addr]
    return state.model_trace[addr]
end

function read(state::FirstPassState, src::AuxInputAddress, ::DiscreteAnn)
    addr = src.addr
    return state.aux_trace[addr]
end

function read(state::FirstPassState, src::AuxInputAddress, ::ContinuousAnn)
    addr = src.addr
    state.results.u_cont_reads[addr] = state.aux_trace[addr]
    return state.aux_trace[addr]
end

function write(state::FirstPassState, dest::ModelOutputAddress, value, ::DiscreteAnn)
    addr = dest.addr
    state.results.update_spec.subspec[addr] = value
    return value
end

function write(state::FirstPassState, dest::ModelOutputAddress, value, ::ContinuousAnn)
    addr = dest.addr
    has_value(state.results.update_spec.subspec, addr) && error("Model address $addr already written to")
    state.results.update_spec.subspec[addr] = value
    state.results.t_cont_writes[addr] = value
    return value
end

function write(state::FirstPassState, dest::AuxOutputAddress, value, ::DiscreteAnn)
    addr = dest.addr
    state.results.u_back[addr] = value
    return value
end

function write(state::FirstPassState, dest::AuxOutputAddress, value, ::ContinuousAnn)
    addr = dest.addr
    has_value(state.results.u_back, addr) && error("Proposal address $addr already written to")
    state.results.u_back[addr] = value
    state.results.u_cont_writes[addr] = value
    return value
end

function copy(state::FirstPassState, src::ModelInputAddress, dest::ModelOutputAddress) 
    from_addr, to_addr = src.addr, dest.addr
    model_choices = get_choices(state.model_trace)
    push!(state.results.t_copy_reads, from_addr)
    if has_value(model_choices, from_addr)
        state.results.update_spec.subspec[to_addr] = model_choices[from_addr]
    else
        set_submap!(state.results.update_spec.subspec, to_addr, get_submap(model_choices, from_addr))
    end
    return nothing
end

function apply_oupm_move(state::FirstPassState, move::OUPMMove)
    state.results.update_spec = UpdateWithOUPMMovesSpec((state.results.update_spec.moves..., move), state.results.update_spec.subspec)
end

function copy(state::FirstPassState, src::ModelInputAddress, dest::AuxOutputAddress)
    from_addr, to_addr = src.addr, dest.addr
    model_choices = get_choices(state.model_trace)
    push!(state.results.t_copy_reads, from_addr)
    if has_value(model_choices, from_addr)
        state.results.u_back[to_addr] = model_choices[from_addr]
    else
        set_submap!(state.results.u_back, to_addr, get_submap(model_choices, from_addr))
    end
    return nothing
end

function copy(state::FirstPassState, src::AuxInputAddress, dest::AuxOutputAddress)
    from_addr, to_addr = src.addr, dest.addr
    push!(state.results.u_copy_reads, from_addr)
    aux_choices = get_choices(state.aux_trace)
    if has_value(aux_choices, from_addr)
        state.results.u_back[to_addr] = aux_choices[from_addr]
    else
        set_submap!(state.results.u_back, to_addr, get_submap(aux_choices, from_addr))
    end
    return nothing
end

function copy(state::FirstPassState, src::AuxInputAddress, dest::ModelOutputAddress)
    from_addr, to_addr = src.addr, dest.addr
    push!(state.results.u_copy_reads, from_addr)
    aux_choices = get_choices(state.aux_trace)
    if has_value(aux_choices, from_addr)
        state.results.update_spec.subspec[to_addr] = aux_choices[from_addr]
    else
        set_submap!(state.results.update_spec.subspec, to_addr, get_submap(aux_choices, from_addr))
    end
    return nothing
end

#####################################################################
# second pass through transform (gets automatically differentiated) #
#####################################################################

struct JacobianPassState{T<:Real}
    model_trace
    aux_trace
    input_arr::AbstractArray{T}
    output_arr::Array{T}
    t_key_to_index::Dict
    u_key_to_index::Dict
    cont_constraints_key_to_index::Dict
    cont_u_back_key_to_index::Dict
end

function read(state::JacobianPassState, src::ModelInputTraceRetValToken, ::DiscreteAnn)
    return get_retval(state.model_trace)
end

function read(state::JacobianPassState, src::AuxInputTraceRetValToken, ::DiscreteAnn)
    return get_retval(state.aux_trace)
end

function read(state::JacobianPassState, src::ModelInputAddress, ::DiscreteAnn)
    addr = src.addr
    return state.model_trace[addr]
end

function read(state::JacobianPassState, src::AuxInputAddress, ::DiscreteAnn)
    addr = src.addr
    return state.aux_trace[addr]
end

function _read_continuous(input_arr, addr_info::Int)
    return input_arr[addr_info]
end

function _read_continuous(input_arr, addr_info::Tuple{Int,Int})
    # TODO to handle things other than vectors, store shape in addr info and reshape?
    (start_idx, len) = addr_info
    return input_arr[start_idx:start_idx+len-1]
 end

function read(state::JacobianPassState, src::ModelInputAddress, ::ContinuousAnn)
    addr = src.addr
    if haskey(state.t_key_to_index, addr)
        return _read_continuous(state.input_arr, state.t_key_to_index[addr])
    else
        return state.model_trace[addr]
    end
end

function read(state::JacobianPassState, src::AuxInputAddress, ::ContinuousAnn)
    addr = src.addr
    if haskey(state.u_key_to_index, addr)
        return _read_continuous(state.input_arr, state.u_key_to_index[addr])
    else
        return state.aux_trace[addr]
    end
end

function write(state::JacobianPassState, dest::ModelOutputAddress, value, ::DiscreteAnn)
    return value
end

function write(state::JacobianPassState, dest::AuxOutputAddress, value, ::DiscreteAnn)
    return value
end

function _write_continuous(output_arr, addr_info::Int, value)
    return output_arr[addr_info] = value
end

function _write_continuous(output_arr, addr_info::Tuple{Int,Int}, value)
    (start_idx, len) = addr_info
    return output_arr[start_idx:start_idx+len-1] = value
end

function write(state::JacobianPassState, dest::AuxOutputAddress, value, ::ContinuousAnn)
    addr = dest.addr
    return _write_continuous(state.output_arr, state.cont_u_back_key_to_index[addr], value)
end

function write(state::JacobianPassState, dest::ModelOutputAddress, value, ::ContinuousAnn)
    addr = dest.addr
    return _write_continuous(state.output_arr, state.cont_constraints_key_to_index[addr], value)
end

function copy(state::JacobianPassState, src, dest)
    return nothing
end

function apply_oupm_move(state::JacobianPassState, move)
    nothing
end

#################################
# computing jacobian correction #
#################################

discard_skip_read_addr(addr, discard::ChoiceMap) = !has_value(discard, addr)
discard_skip_read_addr(addr, discard::Nothing) = false

function store_addr_info!(dict::Dict, addr, value::Real, next_index::Int)
    dict[addr] = next_index 
    return 1 # number of elements of array
end

function store_addr_info!(dict::Dict, addr, value::AbstractArray{<:Real}, next_index::Int)
    len = length(value)
    dict[addr] = (next_index, len)
    return len # number of elements of array
end

function assemble_input_array_and_maps(
        t_cont_reads, t_copy_reads, u_cont_reads, u_copy_reads, rev_update_spec::UpdateWithOUPMMovesSpec)
    assemble_input_array_and_maps(t_cont_reads, t_copy_reads, u_cont_reads, u_copy_reads, rev_update_spec.subspec)
end

function assemble_input_array_and_maps(
        t_cont_reads, t_copy_reads, u_cont_reads, u_copy_reads, discard::Union{ChoiceMap,Nothing})

    input_arr = Vector{Float64}()
    next_input_index = 1

    t_key_to_index = Dict()
    for (addr, v) in t_cont_reads
        if addr in t_copy_reads
            continue
        end
        if discard_skip_read_addr(addr, discard)
            # note: only happens when the model is unchanged
            continue
        end
        next_input_index += store_addr_info!(t_key_to_index, addr, v, next_input_index)
        append!(input_arr, v)
    end

    u_key_to_index = Dict()
    for (addr, v) in u_cont_reads
        if addr in u_copy_reads::DynamicSelection
            continue
        end
        next_input_index += store_addr_info!(u_key_to_index, addr, v, next_input_index)
        append!(input_arr, v)
    end

    return (t_key_to_index, u_key_to_index, input_arr)
end

function assemble_output_maps(t_cont_writes, u_cont_writes)
    next_output_index = 1

    cont_constraints_key_to_index = Dict()
    for (addr, v) in t_cont_writes
        next_output_index += store_addr_info!(cont_constraints_key_to_index, addr, v, next_output_index)
    end

    cont_u_back_key_to_index = Dict()
    for (addr, v) in u_cont_writes
        next_output_index += store_addr_info!(cont_u_back_key_to_index, addr, v, next_output_index)
    end

    return (cont_constraints_key_to_index, cont_u_back_key_to_index, next_output_index-1)
end

function jacobian_correction(transform::OUPMInvolutionDSLProgram, prev_model_trace, proposal_trace, first_pass_results, discard)

    # create input array and mappings input addresses that are needed for Jacobian
    # exclude addresses that were copied explicitly to another address
    (t_key_to_index, u_key_to_index, input_arr) = assemble_input_array_and_maps(
        first_pass_results.t_cont_reads,
        first_pass_results.t_copy_reads,
        first_pass_results.u_cont_reads,
        first_pass_results.u_copy_reads, discard)
    
    # create mappings for output addresses that are needed for Jacobian
    (cont_constraints_key_to_index, cont_u_back_key_to_index, n_output) = assemble_output_maps(
        first_pass_results.t_cont_writes,
        first_pass_results.u_cont_writes)

    # this function is the partial application of the continuous part of the
    # transform, with inputs corresponding to a particular superset of the
    # columns of the reduced Jacobian matrix
    function f_array(input_arr::AbstractArray{T}) where {T <: Real}

        # closing over:
        # - trace, u
        # - u_key_to_index, t_key_to_index, cont_constraints_key_to_index, cont_u_back_key_to_index
        # - proposal_args, proposal_retval

        output_arr = Vector{T}(undef, n_output)

        jacobian_pass_state = JacobianPassState(
            prev_model_trace, proposal_trace, input_arr, output_arr, 
            t_key_to_index, u_key_to_index,
            cont_constraints_key_to_index,
            cont_u_back_key_to_index)

        # mutates the state
        transform.fn!(jacobian_pass_state)

        println("input array:")
        display(input_arr)
        println("output array:")
        display(output_arr)

        # return the output array
        output_arr
    end

    # compute Jacobian matrix of f_array, where columns are inputs, rows are outputs
    J = ForwardDiff.jacobian(f_array, input_arr)
    @assert size(J)[2] == length(input_arr)
    num_outputs = size(J)[1]
    if size(J) != (num_outputs, num_outputs)
        error("Jacobian was not square (size was $(size(J))); the function may not be an bijection")
    end

    # log absolute value of Jacobian determinant
    correction = LinearAlgebra.logabsdet(J)[1]
    if isinf(correction)
        @error "Weight correction is infinite; the function may not be an bijection"
    end
    
    return correction
end

function check_round_trip(trace, trace_rt)
    choices = get_choices(trace)
    choices_rt = get_choices(trace_rt)
    if !isapprox(choices, choices_rt)
        @error("choices: $(sprint(show, "text/plain", choices))")
        @error("choices after round trip: $(sprint(show, "text/plain", choices_rt))")
        error("transform round trip check failed")
    end
    return nothing
end

function check_round_trip(
            prev_model_trace, prev_model_trace_rt,
            forward_proposal_trace, forward_proposal_trace_rt)
    check_round_trip(prev_model_trace, prev_model_trace_rt)
    check_round_trip(forward_proposal_trace, forward_proposal_trace_rt)
    return nothing
end

############################
# SymmetricTraceTranslator #
############################

"""
    apply_oupm_move = OUPMMHKernel(;
        proposal::GenerativeFunction,
        proposal_args::Tuple = (),
        f::OUPMInvolutionDSLProgram)

Construct a function which will apply an OUPM move specified via a proposal + the OUPM involution DSL
to a OUPM trace, and return the new trace and (log) MH acceptance ratio for this move.

Run the translator with:

    (output_trace, log_weight) = apply_oupm_move(input_trace; check=false, observations=EmptyChoiceMap())

Use `check` to enable the involution check (this requires that the transform `f` has been marked with [`is_involution`](@ref)).

If `check` is enabled, then `observations` is a choice map containing the observed random choices, and the check will additionally ensure they are not mutated by the involution.
"""
@with_kw struct OUPMMHKernel
    q::GenerativeFunction
    q_args::Tuple = ()
    f::OUPMInvolutionDSLProgram # an involution
end

function symmetric_trace_translator_run_transform(
        f::OUPMInvolutionDSLProgram,
        prev_model_trace::Trace, forward_proposal_trace::Trace,
        q::GenerativeFunction, q_args::Tuple)
    first_pass_results = run_first_pass(f, prev_model_trace, forward_proposal_trace)
    (new_model_trace, log_model_weight, _, discard) = update(
        prev_model_trace, get_args(prev_model_trace),
        map((_) -> NoChange(), get_args(prev_model_trace)),
        first_pass_results.update_spec, AllSelection())
    log_abs_determinant = jacobian_correction(
        f, prev_model_trace, forward_proposal_trace, first_pass_results, discard)
    println("gonna generate backwards proposal using")
    display(first_pass_results.u_back)
    backward_proposal_trace, = generate(
        q, (new_model_trace, q_args...), first_pass_results.u_back)
    return (new_model_trace, log_model_weight, backward_proposal_trace, log_abs_determinant)
end

function (translator::OUPMMHKernel)(prev_model_trace::Trace; check=false, observations=EmptyChoiceMap())
    # simulate from auxiliary program
    forward_proposal_trace = simulate(translator.q, (prev_model_trace, translator.q_args...,))

    # apply trace transform
    (new_model_trace, log_model_weight, backward_proposal_trace, log_abs_determinant) = symmetric_trace_translator_run_transform(
        translator.f, prev_model_trace, forward_proposal_trace, translator.q, translator.q_args)

    # compute log weight
    forward_proposal_score = get_score(forward_proposal_trace)
    backward_proposal_score = get_score(backward_proposal_trace)
    log_weight = log_model_weight + backward_proposal_score - forward_proposal_score + log_abs_determinant

    if check
        Gen.check_observations(get_choices(new_model_trace), observations)
        forward_proposal_choices = get_choices(forward_proposal_trace)
        (prev_model_trace_rt, _, forward_proposal_trace_rt, _) = symmetric_trace_translator_run_transform(
            translator.f, new_model_trace, backward_proposal_trace, translator.q, translator.q_args)
        check_round_trip(
            prev_model_trace, prev_model_trace_rt,
            forward_proposal_trace, forward_proposal_trace_rt)
    end

    return (new_model_trace, log_weight)
end

function Gen.metropolis_hastings(trace, apply_oupm_move::OUPMMHKernel; check=false, observations=EmptyChoiceMap())
    (new_tr, log_weight) = apply_oupm_move(trace; check=check, observations=observations)
    if log(rand()) <= log_weight
        (new_tr, true)
    else
        (trace, false)
    end
end

export @oupm_involution
export @read, @write, @copy, @tcall
export @birth, @death, @split, @merge, @move
export OUPMInvolutionDSLProgram, OUPMMHKernel