"""
    Call

Represents calling the gen fn with address `addr` on argument `key`
"""
struct Call{addr}
    key
end
Call(addr, key) = Call{addr}(key)
Call(p::Pair{Symbol, <:Any}) = Call(p[1], p[2])
key(call::Call) = call.key
addr(call::Call{a}) where {a} = a
Base.show(io::IO, c::Call) = print(io, "Call($(addr(c) => key(c)))")