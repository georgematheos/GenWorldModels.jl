@gen (static) num_parentless_viruses(::World, ::Tuple{}) = {:num} ~ poisson(3)
@gen (static) num_children(::World, ::Tuple{Virus}) = {:num} ~ poisson(0.9)

@gen (static) is_observed(world, virus::Virus) = {:is_observed} ~ bernoulli(0.7)

@gen function genome(world, virus::Virus)
    origin ~ lookup_or_generate(world[:origin][virus])
    if origin == ()
        genome ~ uniform_random_genome()
        return genome
    else
        parent = origin[1]
        parent_genome ~ lookup_or_generate(world[:genome][parent])
        child_genome ~ genome_mutation_model(parent_genome)
        return child_genome
    end
end

@gen (static, diffs) function phenotype(world, virus::Virus)
    genome ~ lookup_or_generate(world[:genome][virus])
    phenotype ~ sample_phenotype(genome)
    return phenotype
end

@gen (static) function get_observed_viruses_from(world, parent)
    include_this ~ lookup_or_generate(world[:is_observed][parent])
    num_children ~ lookup_or_generate(world[:num_children][parent])
    downstream ~ Map(get_observed_viruses_from)(fill(world, num_children), [Virus((parent,), i) for i=1:num_children])
    itr = Iterators.flatten(downstream)
    possibly_with_parent = include_this ? itr : Iterators.flatten((itr, (parent,)))
    return possibly_with_parent
end

@gen (static) function get_observed_viruses(world)
    num_parentless ~ lookup_or_generate(world[:num_parentless_viruses][()])
    downstream ~ Map(get_observed_viruses_from)(fill(world, num_parentless), [Virus(i) for i=1:num_children])
    return collect(Iterators.flatten(downstream))
end

@gen (static) function get_observations(world)
    #observed_viruses ~ OUPMSet(Virus, :is_observed)(world)
    #phenotypes ~ SetMap(lookup_or_generate)(mgfcall_map(world[:phenotype], observed_viruses))

    observed_viruses ~ get_observed_viruses(world)
    phenotypes ~ Map(lookup_or_generate)(mgfcall_map(world[:phenotype], observed_viruses))

    # our observations are this return value, which "integrates out" the order of the observed phenotypes
    return Set(phenotypes)
end

sample_genomes = UsingWorld(get_observations,
    :phenotype => phenotype,
    :genome => genome,
    :is_observed => is_observed,
    :num_children => num_children,
    :num_parentless_viruses => num_parentless_viruses
)