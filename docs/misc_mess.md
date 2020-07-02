```julia
@type Nonterminal
@type ProductionRule(::Nonterminal)
# turns into
abstract type Nonterminal <: OUPMType end
struct _ID_Nonterminal <: Nonterminal
    id::Identifier
end
struct _IDX_Nonterminal <: Nonterminal
    idx::Int
end

abstract type ProductionRule <: OUPMType end
struct _ID_ProductionRule <: ProductionRule
    id::Identifier
end
struct _IDX_ProductionRule <: ProductionRule
    origin::Tuple{<:Nonterminal}
    idx::Int
end
to_idx_rep(pr::_ID_ProductionRule, world) = _IDX_ProductionRule(
    map(x -> to_idx_rep(x, world), pr.origin),
    get_idx(pr.id, world)
)

idx_origin(pr::_IDX_ProductionRule) = pr.origin
idx_origin(pr::_ID_ProductionRule, world) = to_idx_rep(pr).origin
# internally, the world stores the origin for each ID productionrule in identifier form
```



# expand the given nonterminal to generate text
@gen function generate_text(world, nt::Nonterminal)
    dist_over_prs ~ @w dist_over_prs[nt]
    selected_prod_rule_idx ~ categorical(dist_over_prs)
    selected_prod_rule = ProductionRule(nt, selected_prod_rule_idx)

    next ~ @w production[selected_prod_rule]
    if next isa Terminal
        return next
    elseif next isa Nonterminal
        return {:unary} ~ generate_text(world, next)
    elseif next isa Tuple{Nonterminal, Nonterminal}
        binary1 ~ generate_text(world, next[1])
        binary2 ~ generate_text(world, next[2])
        return Iterators.flatten([binary1, binary2])
    end
end


The core piece of infrastructure
I would like to expose is a primitive update specification for `UsingWorld` called
`MoveSubtraces`.

### `MoveSubtraces`
This update specification tells `UsingWorlds` to change the keys for some subtraces.
We construct it using syntax along the lines of
```julia
MoveSubtraces(
    :mem_gen_fn_addr => [
        current_key => new_key,
        current_key => new key, 
        ...
    ],
    # possibly specify to move subtraces for other memoized generative functions
)
```


The most common inference
algorithm for these models, Markov Chain Monte Carlo, can be viewed as a guided search across the space
of possible assignments to random variables.


### Allowing users to use identifiers outside of memoized generative function keys
So far, as presented, the "identifier trick" allows for performance gains in the case
where the references to the objects whose identifiers are changing are as keys to memoized
generative function calls.  However, we may have references to these objects
within the body of generative functions (kernel or MGF) as well.  In these cases,
some additional reindexing is inevitable without introducing custom distributions
to sample objects in the identifier representation.  Any time we sample an object
using its index, we have a reference to the index we may need to change during reindexing.

However, this necessary reindexing burden can become far worse if a sampled index is passed around
a lot in the body of the generative function (ie. if the "markov blanket" of that index becomes large).
For instance, we might have a function along the lines of:
```julia
@gen function sample_source_and_generate_sounds(world, num_sounds)
    num_sources ~ lookup_or_generate(world[:num_sources][()])
    source_idx ~ uniform_discrete(1, num_sources)
    source = AudioSource(source_idx)
    sounds ~ Map(generate_sound)(fill(source, num_sounds), fill(world, num_sounds))
    # where `generate_sound` is NOT memoized (though it may look up memoized
    # information, like the type of sound for the audio source, in the world)
end
```
In this function, we sample an index for an AudioSource.  But then we pass this source
into a potentially huge number of calls.  If we need to move the index for this audio source,
it's not enough to just change `:source_idx` -- we need to change the arguments for
every call to `generate_sound` made in this function!

To resolve this, the user could do the following:
```julia
@gen function sample_source_and_generate_sounds(world, num_sounds)
    num_sources ~ lookup_or_generate(world[:num_sources][()])
    source_idx ~ uniform_discrete(1, num_sources)

    # use the identifier representation!
    source = identifier_representation(AudioSource(source_idx), world)

    sounds ~ Map(generate_sound)(fill(source, num_sounds), fill(world[:audio_source_type], num_sounds))
    # `generate_sound` is NOT memoized, but it looks up from the memoized generative function `audio_source_type`
end
```
Now, all the calls to `generate_sound` use the identifier representation for the audio source.
When we move an object in the world by changing the mapping between identifiers and objects,
and then change `:source_idx`, the function `identifier_representation` will understand that
this has happened in such a way that the identifier it outputs is unchanged.  (Ie. the world
has had its identifier mapping changed so that the new value of `:source_idx` corresponds to the
old value of )



### TODO: I don't think this example is realistic!  When do we need
### to pass around the audiosource object where we don't pass around the world object?!