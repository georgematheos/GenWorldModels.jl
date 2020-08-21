export ObjectSet

# TODO: should object sets know the type of object they store?
abstract type ObjectSet <: AbstractSet{OUPMObject} end
Base.merge(::ObjectSet, ::ObjectSet) = error("not implemented")
Base.length(::ObjectSet) = error("not implemented")
Base.in(_, ::ObjectSet) = error("not implemented")
Base.iterate(::ObjectSet) = error("not implemented")
Base.iterate(::ObjectSet, _) = error("not implemented")
_uniform_sample(::ObjectSet) = error("not implemented")

struct SimpleObjectSet <: ObjectSet
    objects::PersistentSet
end
SimpleObjectSet(s::Set) = SimpleObjectSet(PersistentSet(s))
Base.merge(s1::SimpleObjectSet, s2::SimpleObjectSet) = SimpleObjectSet(merge(s1.objects, s2.objects))
Base.length(s::SimpleObjectSet) = length(s.objects)
Base.in(obj, s::SimpleObjectSet) = in(obj, s.objects)
Base.iterate(s::SimpleObjectSet) = iterate(s.objects)
Base.iterate(s::SimpleObjectSet, state) = iterate(s.objects, state)
_uniform_sample(s::SimpleObjectSet) = collect(s.objects)[uniform_discrete(1, length(s))]
function Base.show(io::IO, ::MIME"text/plain", set::SimpleObjectSet)
    println(io, "--- Simple Object Set containing $(length(set)) elements ---")
    show(io, set.objects)
    println(io, "\n-------------------------------")
end

struct EnumeratedOriginObjectSet <: ObjectSet
    origin_to_objects::PersistentHashMap
    objects::PersistentSet
end
function EnumeratedOriginObjectSet(origin_to_objects)
    objects = PersistentSet()
    for (origin, set) in origin_to_objects
        for object in set
            objects = push(objects, object)
        end
    end
    return EnumeratedOriginObjectSet(origin_to_objects, objects)
end
function Base.merge(e1::EnumeratedOriginObjectSet, e2::EnumeratedOriginObjectSet)
    println("MERGING $e1 and $e2")
    if length(e1) < length(e2)
        (e2, e1) = (e1, e2)
    end
    o_to_o = e1.origin_to_objects
    objects = e1.objects
    for (origin, set) in e2.origin_to_objects
        if haskey(o_to_o, origin)
            o_to_o = assoc(o_to_o, origin, merge(o_to_o[origin], set))
        else
            o_to_o = assoc(o_to_o, origin, set)
        end
        for obj in e2.objects
            objects = push(objects, obj)
        end
    end
    return EnumeratedOriginObjectSet(o_to_o, objects)
end

Base.length(s::EnumeratedOriginObjectSet) = length(s.objects)
Base.in(obj, s::EnumeratedOriginObjectSet) = in(obj, s.objects)
Base.iterate(s::EnumeratedOriginObjectSet) = iterate(s.objects)
Base.iterate(s::EnumeratedOriginObjectSet, state) = iterate(s.objects, state)
_uniform_sample(s::EnumeratedOriginObjectSet) = collect(s.objects)[uniform_discrete(1, length(s))]
function Base.show(io::IO, m::MIME"text/plain", set::EnumeratedOriginObjectSet)
    println(io, "--- Enumerated Origin Object Set containing $(length(set)) elements for $(length(set.origin_to_objects)) origins ---")
    for (origin, objset) in set.origin_to_objects
        print(io, "Objects for origin ")
        print(io, origin)
        println(io, ": ")
        show(io, m, objset)
        println()
    end
    println(io, "---------------------")
end

include("get_single_origin_object_set.jl")
include("get_origin_iterating_object_set.jl")
export ObjectSet, GetOriginIteratingObjectSet, GetSingleOriginObjectSet