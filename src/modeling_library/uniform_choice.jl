struct UniformChoice <: Gen.Distribution{Any} end
const uniform_choice = UniformChoice()

# TODO: can we avoid collecting the set every time?
function Gen.random(::UniformChoice, set)
    lst = collect(set)
    lst[uniform_discrete(1, length(set))]
end

Gen.logpdf(::UniformChoice, val, set) = val in set ? -log(length(set)) : -Inf
uniform_choice(set) = random(uniform_choice, set)