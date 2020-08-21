import Distributions
import ConjugatePriors

struct NormalInverseGamma <: Gen.Distribution{Tuple{<:Real, <:Real}} end
normal_inverse_gamma = NormalInverseGamma()
Gen.random(::NormalInverseGamma, args...) = rand(ConjugatePriors.NormalInverseGamma(args...))
Gen.logpdf(::NormalInverseGamma, val, args...) = Distributions.logpdf(ConjugatePriors.NormalInverseGamma(args...), val...)
Gen.has_argument_grads(::NormalInverseGamma) = false # TODO: implement these gradients