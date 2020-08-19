const GENOME_LENGTH = 64
const MUTATION_PROB = 0.1

@dist sample_phenotype(genotype) = genotype # TODO: make this stochastic!

struct UniformRandomGenome <: Gen.Distribution end
uniform_random_genome = UniformRandomGenome()
Base.random(::UniformRandomGenome) = [bernoulli(0.5) for _=1:GENOME_LENGTH]
Base.logpdf(::UniformRandomGenome, genome::Vector{Bool}) = length(genome) == GENOME_LENGTH ? -log(GENOME_LENGTH) : -Inf

struct GenomeMutationModel <: Gen.Distribution end
genome_mutation_model = GenomeMutationModel()
function Base.random(::GenomeMutationModel, mutatee::Vector{Bool})
    [
        bernoulli(MUTATION_PROB) ? bernoulli(0.5) : mutatee[i]
        for i=1:GENOME_LENGTH
    ]
end
function Base.logpdf(::GenomeMutationModel, mutated::Vector{Bool}, mutatee::Vector{Bool})
    num_diff = sum(i for i=1:GENOME_LENGTH if mutated[i] !== mutatee[i])
    num_diff * (log(MUTATION_PROB) + log(0.5)) + (GENOME_LENGTH - num_diff) * log(1 - MUTATION_PROB)
end