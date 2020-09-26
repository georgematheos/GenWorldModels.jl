#=
    As a "BLOG" model:

guarantee:
NUM_VIRUS = 1000.
time(Virus[1]) = 0.
genome(Virus[1]) = [some genome].

to generate the other genomes, can run for every other virus:
time(virus) ~ uniform_discrete(1, NUM_VIRUSES)
parent(virus) ~ uniform_from_set({Virus v: time(v) < time(virus)})
genome(virus) ~ mutation_prior(genome(parent(virus)))

Task: given the genome of each virus, can we figure out which virus
evolved from each other virus?

***
other model:
num_children(virus) ~ poisson(1)
genome(virus) ~ mutation_prior(genome(parent(virus)))

all_descendents(virus) ~ [
    child, all_descendents(child)... for child of virus
]
all_genomes ~ map(genome, all_descendents(root_virus))

I have implemented the first of these below.
=#

NUM_VIRUSES = 1000
# genomes represented as vectors of the numbers 1, 2, 3, 4 (ATGC)

@dist time(world, virus) = uniform_discrete(1, NUM_VIRUSES)

@gen function mutation_prior(genome)
    num_changes ~ poisson(2)
    for change in changes
        idx ~ uniform_discrete(length(genome))
        new_val ~ uniform_discrete(1, 4)
        genome = assoc(genome, idx, new_val)
    end
    return genome
end

# @gen (static) genome(world, virus) ~ mutation_prior(~ @w genome[ ~ @w parent[virus] ] )

@gen (static) function genome(world, virus)
    parent ~ @w parent[virus]
    parent_genome ~ @w genome[parent]
    genome ~ mutation_prior(parent_genome)
    return genome
end

# TODO: incremental updates
@gen function time_to_virus(world)
    times = []
    for virus=1:NUM_VIRUSES
        virustime ~ @w time[virus]
        times[virustime] = virus
    end
    return times
end

@gen (static) function parent(world, virus)
    time ~ @w time[virus]
    time_to_virus ~ @w time_to_virus[()]
    parent_time ~ uniform_discrete(1, time - 1)
    parent = time_to_virus[parent_time]
    return parent
end

@gen function get_all_genomes(world)
    genomes = []
    for virus=1:NUM_VIRUSES
        genome ~ @w genome[virus]
        push!(genomes, genome)
    end
    return genomes
end

generate_genomes = UsingWorld(get_all_genomes, :time => time, :genome => genome, :parent => parent)