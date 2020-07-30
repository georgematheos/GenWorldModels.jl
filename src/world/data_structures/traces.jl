"""
    Traces

A data structure containing the subtraces for all the calls in the world.
"""
struct Traces{T}
    traces::T
    function Traces(traces::NamedTuple{A, M}) where {A, M <: Tuple{Vararg{T where T <: PersistentHashMap}}}
        new{NamedTuple{A, M}}(traces)
    end
end

function Traces(MGFAddrs::Tuple{Vararg{Symbol}}, mgfs)
    maps = Tuple(
        PersistentHashMap{Any, Gen.get_trace_type(mgf)}()
        for mgf in mgfs
    )
    Traces(NamedTuple{MGFAddrs}(maps))
end

mgf_addrs(::Traces{NamedTuple{Addrs, MapTypes}}) where {Addrs, MapTypes} = Addrs

function FunctionalCollections.assoc(traces::Traces, call::Call{mgf_addr}, trace::Trace) where {mgf_addr}
    traces_for_addr = traces.traces[mgf_addr]
    new_traces_for_addr = assoc(traces_for_addr, key(call), trace)
    new_traces = merge(traces.traces, NamedTuple{(mgf_addr,)}((new_traces_for_addr,)))
    Traces(new_traces)
end

function FunctionalCollections.dissoc(traces::Traces, call::Call{mgf_addr}) where {mgf_addr}
    traces_for_addr = traces.traces[mgf_addr]
    new_traces_for_addr = dissoc(traces_for_addr, key(call))
    new_traces = merge(traces.traces, NamedTuple{(mgf_addr,)}((new_traces_for_addr,)))
    Traces(new_traces)
end

has_trace(traces::Traces, call::Call) = is_mgf_call(call) && haskey(traces.traces[addr(call)], key(call))
get_trace(traces::Traces, call::Call) = traces.traces[addr(call)][key(call)]
traces_for_mgf(traces::Traces, mgf_addr::Symbol) = traces.traces[mgf_addr]
function all_traces(traces::Traces)
    Iterators.flatten(
        (
            Call(a, key) => trace
            for (key,trace) in traces_for_mgf(traces, a)
        )
        for a in mgf_addrs(traces)
    )
end