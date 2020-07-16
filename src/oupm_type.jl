abstract type OUPMType end
(T::Type{<:OUPMType})(world, idx::Int) = lookup_or_generate_id_object(world, T, idx)