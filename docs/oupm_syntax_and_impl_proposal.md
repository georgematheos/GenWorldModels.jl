## Example Model 1

First, we define a few useful util functions/macros:
```julia
# vector of arguments to a `lookup_or_generate` call
lookup_vector(world::World, addr::Symbol, keys::AbstractVector) = [world[:sound][key] for key in keys]

# map for a memozied generative function, so
# @WorldMap(mem_gen_fn, vector)
# returns [world[mem_gen_fn][val] for val in vector]
macro WorldMap(addr, vect)
    quote Map(lookup_or_generate)(lookup_vector(world, $addr, $vect)) end
end
```

Now we define a little audio model
```julia
##########
# Model #
##########

@type Source # declare the type audio source

### memoized generative functions: ###

# the sound for a given source
@gen (static, diffs) function sound(world, source::Source)
    source_type ~ uniform_discrete(1, NUM_SOURCE_TYPES)
    sound ~ sound_prior_for_type(source_type)
    return sound
end

# we can use distributions as memoized generative functions after my PR to make Dist <: GenFn
@dist is_obstructed(source::Source) = bernoulli(0.2) # whether a given source is obstructed

### non-memoized generative functions: ###

@gen function get_unobstructed_sources(world, sources)
    obstructed_statuses ~ @WorldMap(:is_obstructed, sources)
    unobstructed_statuses = map(!, obstructed_statuses)
    unobstructed_indices = findall(unobstructed_statuses)
    unobstructed_sources = map(idx -> sources[idx], unobstructed_indices)

    return unobstructed_sources
end

@gen (static, diffs) function combined_sound(world, unobstructed_sources)
    # lookup world[:sound][source] for each unobstructed source
    sounds ~ @WorldMap(sound, unobstructed_sources)
    return sum(sounds)
end

# KERNEL:
@gen (static, diffs) function _generate_sound(world)
    num_sources ~ poisson(4)
    sources = map(idx -> Source(idx), collect(1:num_sources))
    unobstructed ~ get_unobstructed_sources(world, sources)
    sound ~ combined_sound(world, unobstructed)
    return sound
end

generate_sound = UsingWorld(_generate_sound, :sound => sound, :is_obstructed => is_obstructed)

#############
# INFERENCE #
#############

@dist dobirth(tr) = bernoulli(0.5)
@dist tobirth(tr) = Source(uniform_discrete(1, tr[:kernel => :num_sources] + 1))
@dist tokill(tr) = Source(uniform_discrete(1, tr[:kernel => :num_sources]))

source_birth_death_mh_kernel = BirthDeathKernel(;
    # generative functions or distributions to decide what addresses to birth/death, and which to do
    do_birth = dobirth,
    to_add = tobirth,
    to_delete = tokill,
    num_statement_addr = source -> :kernel => :num_sources # address of the number statement
)

@kern function audio_inference_kernel(tr)
    # regenerate whether a single source is obstructed
    let source_idx ~ uniform_discrete(1, tr[:kernel => :num_sources])
        tr ~ mh(tr, select(:world => :is_unobstructed => Source(source_idx)))
    end

    # regenerate each sound
    for source in tr[:kernel => :unobstructed]
        tr ~ mh(tr, select(:world => :sound => source))
    end

    # add or delete a source
    tr ~ source_birth_death_mh_kernel(tr)
end
```

## Explanation

The "newest" thing in this example model is the introduction of the `@type` syntax,
and the construction of a `BirthDeathKernel`.  I will elaborate on how these work.

First, let's investigate the birth/death moves, since this will motivate the `@type` syntax.
To make it easy to do simple birth/death moves, we expose a constructor `BirthDeathKernel`
which returns a MH kernel for a birth/death move.  When constructing this kernel, we have
a lot of flexibility.  We specify:
1. A distribution, accepting the trace as argument, proposing whether to do a birth or death move.
2. A distribution, accepting the trace as argument, proposing what object to birth, if this is a birth move.
3. A distribution, accepting the trace as argument, proposing what object to delete, if this is a death move.
4. A function of the object to add or delete, returning the address in the model which contains the "number statement"
    this object came from.  This number statement will be increased during a birth and decreased during a death.

If it is too constraining to use distributions rather than generative functions to decide what objects
to birth/death, we could instead construct the `BirthDeathKernel` by giving a custom generative function
to determine whether to do the birth or death, and which objects to birth/death, along with an involution
for the choices from this generative function to get the reverse-direction choices.  This would look like:

```julia
@gen function determine_move(tr)
    if {:do_birth} ~ bernoulli(0.5)
        birth_idx ~ uniform_discrete(1, tr[:kernel => :num_sources] + 1)
        return Source(birth_idx)
    else
        death_idx ~ uniform_discrete(1, tr[:kernel => :num_sources])
        return Source(death_idx)
    end
end
@involution function get_reverse_move()
    do_birth = @read_discrete_from_proposal(:do_birth)
    @write_discrete_to_proposal(:do_birth, !do_birth)
    if do_birth
        @write_discrete_to_proposal(:birth_idx, @read_discrete_from_proposal(:death_idx))
    else
        @write_discrete_to_proposal(:death_idx, @read_discrete_from_proposal(:birth_idx))
    end
end

source_birth_death_kernel = BirthDeathKernel(;
    proposal = determine_move,
    get_reverse_move = get_reverse_move,
    num_statement_addr = source -> :kernel => :num_sources
)
```
This allows for users to specify which objects to birth/death in a more complex way,
but still handles the bookkeeping of updating number statements, and moving around random choices as needed.

### What bookkeeping does this kernel do behind-the-scenes?

TODO: expand upon what's happening here.
Short version: Changes the number statement, shifts over the other indices to make room/lower everything.
Since we're using indices, all the references to things end up in the right places.
For birth move, by default, generates all the choices for the new object.
For the death move, doesn't need to generate any choices ofc, but still passes in the proper "selection"
into the update function so that the internal proposal distribution gives us the regenerate score,
to reflect that these choices are made by the internal dist.
It knows which choices to constrain by simply selecting every call in the world for a MGF
which takes `Source` as an input.

TODO: add a way where we can specify how to change some of the choices using a regular involution.

## Identifiers and "types"

When doing inference moves to change the number of objects, as part of the bookkeeping we
frequently need to "reindex" some objects.  For instance, in the ongoing example, if I
delete `Source(1)`, I need to reindex sources 1 through n, ie. moving all choices for
`Source(i)` to not be a choice for `Source(i-1)`.  If there are many choices for each object,
this might be a very expensive operation.  It is `O(N * C)`, where `N` is the number
of objects which must be reindexed, and `C` is the number of choices per object.
The "identifier" solution reduces this to be `O(N)`.  To get to `O(1)` we would need
to change the semantics to use "abstract partial instantiations", or something analagous.

The "identifier" trick works as follows.
First, for every object `Obj(i)`, we
select an identifier `ID(Obj(i))`, and fill in a 2-way hashmap between objects and their IDs.
Then, every time we generate a choice for `Obj(i)` by calling a memoized generative function
on `Obj(i)`, `UsingWorld` swaps this out for `ID(Obj(i))`.  So the memoized generative function
in fact just has record of the ID which was used.  Then, if we need to reindex objects 1 through M,
we simply change the ID associated with each of these objects.  In effect, this "shifts the choices",
which are addressed by the IDs.

To make this valid, we want the fact that this swap is happening to be close to invisible to the users.
From the user's perspective, they should not be able to tell the difference between if `UsingWorld` is swapping
in the identifiers, or is using the object indices.

To accomplish this, for types of objects we want to use this "identifier" trick for,
we create a custom type of object.  This is done by the `@type` syntax.
In fact, we create two different types of objects when we do this:

```julia
@type Source
# macroexpands into:

abstract type Source <: OUPMType end
struct _ID_Source <: Source
    id::Identifier
end
struct _IDX_Source <: Source
    idx::Int
end
Source(idx::Int) = _IDX_Source(idx)
function Source(idx::Int, world::World)
    id = lookup_or_generate_object_id!(world, Source(idx))
    _ID_Source(id)
end
to_idx_rep(s::_ID_Source, world::World) = Source(get_idx(s.id))
to_id_rep(s::_IDX_Source, world::World) = Source(s, world)
```

Whenever we call `lookup_or_generate` on one of these custom objects,
it is automatically converted into the identifier representation:
```julia
sound ~ lookup_or_generate(world[:sound][Source(idx)])
# behaves identically to
sound ~ lookup_or_generate(world[:sound][Source(idx, world)])
```

And the choicemap for UsingWorld will automatically transform the keys
with identifiers back into keys with indices, so for the most part the user never
sees the identifiers.

#### Minor issue with this approach

This is not perfect, in the sense that there are cases where identifier objects will appear different
than indexed objects to users.

While the UsingWorld chociemap will automatically convert the lookup keys for memoized generative functions
to the indexed representation, it does not do this for identifier objects which appear in choices within
generative functions.  In this case, users can convert the identifier representation to the index
representation using the function `to_idx_rep`.

I don't think this imperfection is a deal-breaker; while it would be nice to totally hide this from the user,
I don't think it introduces the possibility for hidden errors, and I don't think it is too unclear what's going on;
all this will be documented.  I think it will in general be important for users to understand this distinction
between identifiers and objects, since it can change the meaning of updates somewhat.

## Objects with Origins

It can be useful to have these objects have an "origin", similar to in BLOG.
To illustrate this, here's a simple example that generates a PCFG, and then
some text according to that PCFG.

```julia
@type Nonterminal

# declare the type ProductionRule; each ProdRule has a Nonterminal as its origin
@type ProductionRule(::Nonterminal)

@dist num_nonterminals(world, ::Tuple{}) = poisson(40)
@dist num_production_rules(world, nt::Nonterminal) = poisson(3)

# what do we produce for a given production rule?
@gen function production(world, pr::ProductionRule)
    is_binary_rule ~ bernoulli(0.6)
    if !is_binary_rule
        is_terminal ~ bernoulli(0.5)
        if is_terminal
            return {:terminal} ~ sample_terminal(world)
        else
            num_nts ~ @w num_nonterminals[()]
            next_nt_idx ~ uniform_discrete(1, num_nts)
            return Nonterminal(next_nt_idx)
        end
    else
        # ...generate a binary rule...
        return (nonterminal1, nonterminal2)
    end
end

# get the categorical distribution over a the production rules indices for a nonterminal
@gen (static, diffs) function dist_over_prs(world, nt::Nonterminal)
    num_prs ~ @w num_production_rules[nt]
    return {:dist} ~ dirichlet(fill(DIRICHLET_PRIOR, num_prs))
end

@gen function prs_for_nonterminal(world, nt)
    num_prs ~ @w num_production_rules[nt]
    prs = [Nonterminal(nt, i, world) for i=1:num_prs]
    return {:prs} ~ @WorldMap(:production, prs)
end

@gen (static, diffs) function _generate_pcfg_and_sentences(world, num_sentences)
    num_nts ~ @w num_nonterminals[()]
    nts = [Nonterminal(i, world) for i=1:num_nts]
    pr_dists ~ @WorldMap(:dist_over_prs, nts)
    prs_for_each_nonterminal ~ Map(prs_for_nonterminal)(fill(world, num_nts), nts)
    prs = Set([nt => pr for (nt, prs) in zip(nts, prs_for_each_nonterminal) for pr in prs])
    start_nt = Nonterminal(1, world)

    pcfg = PCFG(nts, pr_dists, prs, start_nt)

    # `generate_text` is a custom generative function / distribution which calculates
    # probabilities using the inside/outside parsing algorithm
    sentences ~ Map(generate_text)(fill(pcfg, num_sentences))
    return sentences
end

generate_pcfg_and_sentences = UsingWorld(
    _generate_pcfg_and_sentences,
    :num_nonterminals=num_nonterminals,
    :num_production_rules=num_production_rules,
    :production=production,
    :dist_over_prs=dist_over_prs
)
```

If we want to do birth/death over the production rules, we could use the following kernel:
```julia
@dist dobirth(tr) = bernoulli(0.5)
@gen function proposal(tr)
    nt_idx ~ uniform_discrete(1, tr[:world => :num_nonterminals => ()]) # NT we will add or delete a PR for
    nt = Nonterminal(nt_idx)
    if {:do_birth} ~ bernoulli(0.5)
       pr_idx ~ uniform_discrete(1, tr[:world => :num_production_rules => nt] + 1)
       dist_over_prs ~ # regenerate the distribution to have 1 more element
    else
       pr_idx ~ uniform_discrete(1, tr[:world => :num_production_rules => nt])
       dist_over_prs ~ # regenerate the distribution to have 1 fewer element
    end
    return ProductionRule(nt, pr_idx)
end
@involution function pr_birth_death_involution()
    @copy_proposal_to_proposal(:nt)
    @copy_proposal_to_proposal(:pr_idx)
    @write_discrete_to_proposal(:do_birth, !@read_discrete_from_proposal(:do_birth))

    # update the distribution over the production rules
    nt = @read_discrete_from_proposal(:nt)
    model_pr_dist = @read_continuous_from_model(:world => :dist_over_prs => nt)
    proposal_dist_pr = @read_continuous_from_proposal(:dist_over_prs)
    @write_continuous_to_proposal(:dist_over_prs, model_pr_dist)
    @write_continuous_to_model(:world => :dist_over_prs => proposal_pr_dist) 
end

source_birth_death_mh_kernel = BirthDeathKernel(;
    # generative functions or distributions to decide what addresses to birth/death, and which to do
    proposal = proposal,
    involution = pr_birth_death_involution,
    num_statement_addr = (prod_rule, world) -> :world => :num_production_rules => origin(prod_rule, world)[1]
)
```

Here, it is useful for performance to have the notion of an "origin" for each production rule that `UsingWorld`
is aware of, since this means `UsingWorld` can heirarchically index these production rules.  This way, when we
birth a PR for one nonterminals, we don't have to reindex the production rules for any other nonterminal.
We might have hundreds of total production rules, but so long as we only have a few for each nonterminal,
the reindexing will never have to move very many PRs at once.

Another move we might want to do is Split/Merge over the nonterminals.
A simple version of this might look like:
```julia
@dist random_nonterminal(num_nonterminals) = Nonterminal(uniform_discrete(1, num_nonterminals))
@dist function uniform_except_value(val, maxval)
    idx ~ uniform_discrete(1, maxval - 1)
    if idx >= val
        return idx + 1
    else
        return idx
    end
end
@dist random_nt_except(num_nts, nt) = Nonterminal(uniform_except_value(nt.idx, num_nts))

@gen function proposal(tr)
    do_split ~ bernoulli(0.5)
    num_nts = tr[:world => :num_nonterminals => ()]
    if do_split
        to_split ~ random_nonterminal(num_nts)
        split_idx1 ~ uniform_discrete(1, num_nts + 1)
        split_idx2 ~ uniform_except_value(split_idx1, num_nts + 1)
    else
        to_merge1 ~ random_nonterminal(num_nts)
        to_merge2 ~ random_nt_except(num_nts, to_merge1)
        merge_idx ~ uniform_discrete(1, num_nts - 1)
    end
end

@dist random_split(a, b) = bernoulli(0.5) ? a : b
@gen function child_proposal(tr, to_split, split_idx1, split_idx2)
    # this function is responsible for deciding which object that have `to_split` as the origin
    # go to new object 1 vs new object 2
    for i=1:tr[:world => :num_production_rules => :to_split]
        {ProductionRule(i, to_split)} ~ random_split(split_idx1, split_idx2)
    end
end
@gen function choices_proposal(tr, to_split, split_idx1, split_idx2)
    # this function is responsible for deciding, for all the choices returning `to_split`,
    # which of the new objects it should go to
    for production_rule in all_prs_with_next(tr)
        addr = :world => :production => production_rule => :next_nt_idx
        if tr[addr] == to_split
            {addr} ~ random_split(split_idx1, split_idx2)
        end
    end
end

function all_prs_with_next(tr)
    return (
        production_rule
        for production_rule in ...
        for nonterminal in ...
        if has_value(get_submap(get_choices(tr), :world => :production => production_rule => :next_nt_idx))
    )
end

SplitMergeMHKernel(;
    proposal = proposal,
    num_addr = (nt, world) -> :world => :num_nonterminals => (),

    # this function returns an iterator over addresses which need to be reindexed if the indices
    # have been shifted by the split/merge and are not the addresses being split/merged
    get_choices_to_reindex = all_prs_with_next,

    # moves needed so the objects being split end up being chosen in the right places
    choices_proposal = choices_proposal,

    # moves needed to divy the children between the new split values
    children_proposal = children_proposal
)
```