struct UniformChoice <: Gen.Distribution{Any} end
uniform_choice = UniformChoice()

function Gen.get_choices(tr::Gen.DistributionTrace{<:Any, UniformChoice})
    world = get_args(tr)[1]
    v = get_retval(tr)
    Value(GenWorldModels.convert_to_concrete(world, v))
end

function Gen.random(::UniformChoice, ::World, s::Set)
    # println("random called with set")
    # display(s)
    idx = uniform_discrete(1, length(s))
    return collect(s)[idx]
end
function Gen.logpdf(::UniformChoice, obj, ::World, s)
    if obj in s
        -log(length(s))
    else
        # println("TRIED SAMPLING $s FROM SET WHICH DOES NOT CONTAIN IT!")
        -Inf
    end
end

function Gen.generate(g::UniformChoice, (world, s)::Tuple, v::Value{<:ConcreteIndexOUPMObject})
    abst = GenWorldModels.convert_to_abstract(world, get_value(v))
    return Gen.generate(g, (world, s), Value(abst))
end


function Gen.update(tr::Gen.DistributionTrace{<:Any, UniformChoice}, (world, s)::Tuple, argdiffs::Tuple, v::Value{<:ConcreteIndexOUPMObject}, sel::AllSelection)
    _converting_update(tr, (world, s), argdiffs, v, sel)
end
function Gen.update(tr::Gen.DistributionTrace{<:Any, UniformChoice}, (world, s)::Tuple, argdiffs::Tuple, v::Value{<:ConcreteIndexOUPMObject}, sel::EmptySelection)
    _converting_update(tr, (world, s), argdiffs, v, sel)
end

function _converting_update(tr::Gen.DistributionTrace{<:Any, UniformChoice}, (world, s), argdiffs, v, sel)
    try
        abst = GenWorldModels.convert_to_abstract(world, get_value(v))
        return Gen.update(tr, (world, s), argdiffs, Value(abst), sel)
    catch e
        println("in converting update with $(get_value(v))")
        display(world.id_table)
        println(("Is there a value for the truth of $(get_value(v))? : ", GenWorldModels.has_val(world, GenWorldModels.Call(:num_facts, get_value(v).origin))))
        println("Is $(get_value(v)) true? : ", lookup_or_generate(world[:num_facts][get_value(v).origin]))
        throw(e)
    end
end