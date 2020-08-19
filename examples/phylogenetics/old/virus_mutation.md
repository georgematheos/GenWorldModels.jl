(put online for sharing at [this gist](https://gist.github.com/georgematheos/8de66b5e96997cad81f4ceb533fe1b8f))

# `UsingWorld` example: Inferring Phylogenic Trees

One practical way MCMC is used in biology is for inferring phylogenic trees.
This means that the an MCMC program is used to infer the evolutionary relationships
between different species, ie. what organism did each organism evolve from?
This is done by comparing the genomes of different species.
[Here's a little paper about this; I have not read it fully in-depth.](https://www.sciencemag.org/site/feature/data/1050262.pdf)

### Simple Example Case: Modeling Virus Mutation

For the sake of a simple example, we will say our task is to
build phylogenic trees to understand how different viruses evolved into one another.
Our goal will be to have a program with the following interface:

*Input*:
- A list of virus genomes from distinct viruses
- An assertion about how many viruses there are whose genome we do not know.

*Output*:
- The most probable phylogenic tree of the viruses from the input.  (Ie.
  for each virus, the virus genome it was most likely
  to have evolved from--or that it was most likely to have evolved from
  a specific virus whose genome we have not observed.)

### Simple closed-universe model:
Say that there are `NUM_VIRUSES` in total, of which
we have observed the genomes of `NUM_KNOWN_GENOMES < NUM_VIRUSES`.
We posit the following probabilistic model, written in a BLOG-like syntax:

```
time_evolved(virus) ~ uniform_discrete(1, NUM_VIRUSES)
parent(virus) ~ uniform_from_set({v : time_evolved(v) < time_evolved(virus)})
genome(virus) ~ genome_mutation_model(genome(parent(virus)))
```

One issue with this model is that if every virus has a parent, and there
are no circular parent relationships (where A is the parent of B and B is the parent of A),
then there must be an infinite number of viruses!  We don't want this, so we posit
that there is a "Adam/Eve virus" which has no parents, and whose genome is
sampled from a distribution that does not depend on any parents.
We add to the model:
```
genome(Virus(1)) ~ uniform_over_possible_genomes()
time_evolved(Virus(1)) = 1
```

We label the viruses we know the genome for `2` through `NUM_KNOWN_GENOMES + 1`.
Our observations are:
- For `i=1:NUM_KNOWN_GENOMES`, `genome(Virus(i+1)) = input[i]`

#### Why do I include `time_evolved` in the model?

For the semantics of a `UsingWorld` to be valid, any instantiation
of random variables which depend upon each other in a cyclic way must have 0 probability.
(Milch goes into some detail about this in his thesis; he describes how we want
the model to have a "supportive numbering"--ie. the random variables in the world should
have a valid topological ordering in every state which has nonzero pdf.)

If we were to sample the parent of each virus uniformly from the set of all viruses, there
would be nonzero probability of ending up in a situation where
```
parent(Virus(1)) = Virus(2)
parent(Virus(2)) = Virus(1)
```
This would mean our model is not semantically valid, because we would then could have, with nonzero probability:
```
genome(Virus(1)) ~ virus_mutation_model(genome(Virus(2)))
genome(Virus(2)) ~ virus_mutation_model(genome(Virus(1)))
```
How would one perform `generate` in such a model?  It's unclear, and the semantics I have
currently written from `UsingWorld` require that this situation can never arise.

To ensure that the parent graph is always a DAG in our model, I introduce the idea
of the `time_evolved` for each virus, which can intuitively be thought of as the time
at which each virus species first evolved from its parent.  By requiring the parent
relationships to respect this ordering, we can never have a cyclic dependency graph, and
therefore the model becomes semantically valid.

### Inference:
We will begin with an initial state in which every virus has `Virus(0)` as its parent,
and has the time correponding to its index.
We will have 3 simple MH moves:
1. Select a random virus; regenerate its parent.
2. Select 2 random viruses, A and B, and swap their `time_evolved`.
    (Note that this will often result in a 0 probability state and be rejected.)
3. Select some number of viruses whose genomes we have not observed; regenerate their genomes.  Maybe also do something
    involving changing the `time_evolved` for some of these...basically to further illustrate what a complicated update
    `UsingWorld` can handle.
   (I need to hash this out--the idea is that we might get a better probability by updating
   several genomes at once, eg. to insert a "smooth transition" between 2 observed genomes.
   Another idea would be to insert a chain of a few non-observed viruses between 2 viruses
   and then generate genomes resulting in a "smooth transition"; we might have to change
   some `time_evolved` values to do this in a valid way.  Anyway, the point is really
   to show off that we can do this sort of thing with `UsingWorld`.)

### Why is this a good example of the benefits of UsingWorld?
This MCMC algorithm demonstrates a few features of `UsingWorld`:
1. We don't waste time instantiating genomes which are not relevant in our phylogeny tree.
    (Ie. we don't have a value in the world for genomes we don't think are ancestors of any
    observed data points.)
2. We have many hard-to-predict changes to the dependency graph, since we are exploring different
    trees of gene evolution.
3. Update `3` requires having a topological order available since it performs several updates at once
   and the system needs to know which update to do first.

Another benefit is that it is easy to see how to turn this into an open-universe model:
we make the number of unobserved genes variable.

Another benefit?  I hear virus modeling is a hot topic these days!

#### A possible issue with the example

One thing that occurs to me is that if we have this `time_evolved` available, it's almost
like we're doing manual topological-order tracking.  I think it is still convenient and useful
to have `UsingWorld` automatically calculating the order it needs to update random variables
in an update which modifies multiple RVs.  However, critics could argue that this automatic
tracking is not very important, since we could just specify the order to update RVs every
time we call the `update` function, by using `time_evolved`.
I would argue that it would be a real hassle to have to specify the order these updates should be
performed in each time, and that this increases cognitive load on the programmer nontrivially,
but I don't know how convincing this argument would be to those critics.

### A note on the style of generative model
When I described the probability distribution above, I do describe it as a sort of
generative process, in which a parent is generated for each virus.
I think this is a pretty good model for this situation, since it
very explicitly represents the thing we are interested in: the parent of each virus.
However, I recognize that one could argue that it is more "generative" in style to model a process
in which each virus evolves into some number "children" viruses with a modified genome
Such a model would look something more like this in BLOG-like syntax:

```
Type Virus;
guarenteed Virus Virus0;

# for each virus, there are some number of children; the number of children is sampled from a poisson
Origin Parent(Virus)::Virus;
#Virus(Parent=virus) ~ poisson(0.99);

genome(virus) ~ if virus == Virus0
                    ~ uniform_prior_over_genomes()
                else
                    ~ mutation_model(genome(Parent(virus)));

virus_samples ~ sample_without_replacement_uniform({Virus v}, NUM_KNOWN_GENOMES);
for i=1:NUM_KNOWN_GENOMES
    observe genome(virus_samples[i]) = KNOWN_GENOMES[i];
end
```

We could also implement this with `UsingWorld`, but I think it's a little bit harder than the first model.
Things that might be a bit harder:
- Getting `sample_without_replacement_uniform` to work efficiently
- For efficient inference, I will probably need to do the sort implement "identifier" trick Alex and I discussed

In this model, we might also want to run split/merge or birth/death inference over viruses,
which would take some work to implement.

It sort of appeals to me that there are these 2 different ways to write the model,
one of which I feel `UsingWorld` is already pretty well suited for,
and the other of which I am thinking the next part of the open universe research
should make easier to work with.
