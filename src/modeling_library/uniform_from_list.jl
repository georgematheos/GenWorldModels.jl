struct UniformFromList <: Gen.Distribution{Any} end
const uniform_from_list = UniformFromList()
Gen.random(::UniformFromList, list) = list[uniform_discrete(1, length(list))]
Gen.logpdf(::UniformFromList, obj, list) = obj in list ? -log(length(list)) : -Inf