abstract type ObjectSet end
Base.length(::ObjectSet) = error("not implemented")
Base.in(_, ::ObjectSet) = error("not implemented")
Base.iterate(::ObjectSet) = error("not implemented")
Base.iterate(::ObjectSet, _) = error("not implemented")
sample_uniform(::ObjectSet) = error("not implemented")

struct SimpleObjectSet <: ObjectSet
    objects::PersistentSet
end
Base.length(s::SimpleObjectSet) = length(s.objects)
Base.in(obj, s::SimpleObjectSet) = in(obj, s.objects)
Base.iterate(s::SimpleObjectSet) = iterate(s.objects)
Base.iterate(s::SimpleObjectSet, state) = iterate(s.objects, state)
sample_uniform(s::SimpleObjectSet) = collect(s.objects)[uniform_discrete(1, length(s))]

struct EnumeratedOriginObjectSet <: ObjectSet
    size::Int
    origin_to_objects::PersistentHashMap
    objects::PersistentSet
end
Base.length(s::EnumeratedOriginObjectSet) = s.size
Base.in(obj, s::EnumeratedOriginObjectSet) = in(obj, s.objects)
Base.iterate(s::SimpleObjectSet) = iterate(s.objects)
Base.iterate(s::SimpleObjectSet, state) = iterate(s.objects, state)
sample_uniform(s::SimpleObjectSet) = collect(s.objects)[uniform_discrete(1, length(s))]
