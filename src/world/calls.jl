const _world_args_addr = :args

"""
    Calls

A data structure containing the subtraces for all the calls in the world.

(Some of the calls stored may not be represented using traces.
For instance, "world args" are stored in this data structure as call nodes
which are not associated with any trace.)
"""
struct Calls{T, U}
    traces::T
    args::U
    function Calls(traces::NamedTuple{A, M}, args::NamedTuple{ArgAddrs, ArgTypes}) where {ArgAddrs, ArgTypes, A, M <: Tuple{Vararg{T where T <: PersistentHashMap}}}
        new{NamedTuple{A, M}, NamedTuple{ArgAddrs, ArgTypes}}(traces, args)
    end
end

function Calls(MGFAddrs::Tuple{Vararg{Symbol}}, mgfs, args::NamedTuple)
    maps = Tuple(
        PersistentHashMap{Any, Gen.get_trace_type(mgf)}()
        for mgf in mgfs
    )
    Calls(NamedTuple{MGFAddrs}(maps), args)
end

mgf_addrs(::Calls{NamedTuple{Addrs, MapTypes}, NamedTuple{ArgAddrs, ArgTypes}}) where {Addrs, MapTypes, ArgAddrs, ArgTypes} = Addrs
get_args(calls::Calls) = calls.args

function FunctionalCollections.assoc(calls::Calls, call::Call{mgf_addr}, trace::Trace) where {mgf_addr}
    traces_for_addr = calls.traces[mgf_addr]
    new_traces_for_addr = assoc(traces_for_addr, key(call), trace)
    new_traces = merge(calls.traces, NamedTuple{(mgf_addr,)}((new_traces_for_addr,)))
    Calls(new_traces, calls.args)
end

function FunctionalCollections.dissoc(calls::Calls, call::Call{mgf_addr}) where {mgf_addr}
    traces_for_addr = calls.traces[mgf_addr]
    new_traces_for_addr = dissoc(traces_for_addr, key(call))
    new_traces = merge(calls.traces, NamedTuple{(mgf_addr,)}((new_traces_for_addr,)))
    Calls(new_traces, calls.args)
end

function change_args_to(calls::Calls{T, NamedTuple{ArgAddrs, ArgTypes}}, new_args::NamedTuple{ArgAddrs, <:Tuple}) where {T, ArgAddrs, ArgTypes}
    Calls(calls.traces, new_args)
end

@generated function get_val(calls::Calls, call::Call{mgf_addr}) where {mgf_addr}
    if mgf_addr == _world_args_addr
        quote calls.args[key(call)] end
    else
        quote get_retval(calls.traces[$(QuoteNode(mgf_addr))][key(call)]) end
    end
end
@generated function has_val(calls::Calls, call::Call{mgf_addr}) where {mgf_addr}
    if mgf_addr == _world_args_addr
        quote haskey(calls.args, key(call)) end
    else
        quote haskey(calls.traces[$(QuoteNode(mgf_addr))], key(call)) end
    end
end

get_trace(calls::Calls, call::Call) = calls.traces[addr(call)][key(call)]
traces_for_mgf(calls::Calls, mgf_addr::Symbol) = calls.traces[mgf_addr]
function all_traces(calls::Calls)
    Iterators.flatten(
        (
            Call(a, key) => trace
            for (key,trace) in traces_for_mgf(calls, a)
        )
        for a in mgf_addrs(calls)
    )
end

is_mgf_call(c::Call) = addr(c) !== _world_args_addr