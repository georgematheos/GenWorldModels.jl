# the type for call addresses
const CallAddr = Symbol

"""
    Call

Represents calling the gen fn with address `addr` on argument `key`
"""
struct Call{addr, KeyType}
    key::KeyType
end
Call(addr, key::T) where {T} = Call{addr, T}(key)
Call(p::Pair{<:CallAddr, <:Any}) = Call(p[1], p[2])
key(call::Call) = call.key
addr(call::Call{a}) where {a} = a
Base.show(io::IO, c::Call) = print(io, "Call($(addr(c) => key(c)))")