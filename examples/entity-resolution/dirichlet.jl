import Distributions
struct Dirichlet <: Distribution{Vector{<:Real}} end
const dirichlet = Dirichlet()
Gen.random(::Dirichlet, α) = rand(Distributions.Dirichlet(α))
Gen.logpdf(::Dirichlet, v, α) = Distributions.logpdf(Distributions.Dirichlet(α), v)
dirichlet(α) = random(dirichlet, α)