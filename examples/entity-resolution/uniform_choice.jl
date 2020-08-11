struct UniformChoice <: Gen.Distribution{Any} end
uniform_choice = UniformChoice()

function Gen.random(::UniformChoice, ::World, s::Set)
    println("random called with set")
    display(s)
    idx = uniform_discrete(1, length(s))
    return collect(s)[idx]
end
function Gen.logpdf(::UniformChoice, obj, ::World, s)
    if obj in s
        -log(length(s))
    else
        -Inf
    end
end

function Gen.generate(g::UniformChoice, (world, s)::Tuple, v::Value{<:ConcreteIndexOUPMObject})
    abst = GenWorldModels.convert_to_abstract(world, get_value(v))
    return Gen.generate(g, (world, s), Value(abst))
end