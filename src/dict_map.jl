#=
Extra `generate` and `update` method for `DictMap` to ensure they work when we pass in a `world`
object so we can automatically convert constraints to abstract form.

Ie. these definitions make it so that if one has a call `~ DictMap()(world, dict)`,
then every constraint with a concrete object as address will be converted to being a abstract object.
=#

function Gen.generate(dm::Gen.DictMap, (world, dict)::Tuple{<:World, <:AbstractDict}, constraints::ChoiceMap)
    generate(dm, (dict,), to_abstract_repr(world, constraints, depth=1))
end

function Gen.update(tr::Gen.DictTrace, (world, dict)::Tuple{<:World, <:AbstractDict}, (_, diff)::Tuple{<:Diff, <:Union{NoChange, <:DictDiff}}, spec::UpdateSpec, eca::Selection)
    update(tr, (dict,), (diff,), to_abstract_repr(world, spec, depth=1), to_abstract_repr(world, eca, depth=1))
end