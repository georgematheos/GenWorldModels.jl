using Gen;
using Distributions;

### Computation of likelihood which scales linearly 
struct NoisyMatrix <: Gen.Distribution{Matrix{Float64}} end

const noisy_matrix = NoisyMatrix()

function Gen.logpdf(::NoisyMatrix, x::Matrix{Float64}, mu::Matrix{Float64}, noise::Float64)
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function Gen.random(::NoisyMatrix, mu::Matrix{Float64}, noise::Float64)
    mat = copy(mu)
    (w, h) = size(mu)
    for i=1:w, j=1:h
        mat[i, j] = mu[i, j] + randn() * noise
    end
    mat
end
Gen.has_output_grad(::NoisyMatrix) = false
Gen.has_argument_grads(::NoisyMatrix) = (false, false)
Gen.logpdf_grad(::NoisyMatrix, x, mu, noise) = (nothing, nothing, nothing)

"""
    TruncatedExponential(p::Real, u::Real)
Sample an `Real` from the Truncated distribution with parameter `p` and upper limit u.
"""
struct TruncatedExponential <: Gen.Distribution{Float64} end

const truncated_exponential = TruncatedExponential()

function Gen.logpdf(::TruncatedExponential, x::Real, p::Real, u::Real)
    Distributions.logpdf(Distributions.Truncated(Distributions.Exponential(p), 0, u), x)
end

function Gen.random(::TruncatedExponential, p::Real, u::Real)
    rand(Distributions.Truncated(Distributions.Exponential(p), 0, u)) 
end

Gen.has_output_grad(::TruncatedExponential) = false
Gen.has_argument_grads(::TruncatedExponential) = (false,false)
Gen.logpdf_grad(::TruncatedExponential, x, p, u) = (nothing, nothing, nothing)

"""
    TruncatedGamma(a::Real,b::Real,u::Real)
Sample an `Real` from the Truncated distribution with shape/scale a/b and upper limit u.
"""
struct TruncatedGamma <: Gen.Distribution{Float64} end

const truncated_gamma = TruncatedGamma()

function Gen.logpdf(::TruncatedGamma, x::Real, a::Real, b::Real, u::Real)
    Distributions.logpdf(Distributions.Truncated(Distributions.Gamma(a, b), 0, u), x)
end

function Gen.random(::TruncatedGamma, a::Real, b::Real, u::Real)
    rand(Distributions.Truncated(Distributions.Gamma(a, b), 0, u)) 
end

Gen.has_output_grad(::TruncatedGamma) = false
Gen.has_argument_grads(::TruncatedGamma) = (false,false)
Gen.logpdf_grad(::TruncatedGamma, x, a, b, u) = (nothing, nothing, nothing)

"""
    LogNormal(mu::Real,sig::Real,u::Real)
Sample an `Real` from the Log Normal with mu and sig
"""
struct LogNormal <: Gen.Distribution{Float64} end

const log_normal = LogNormal()

function Gen.logpdf(::LogNormal, x::Real, mu::Real, sig::Real)
    Distributions.logpdf(Distributions.LogNormal(mu, sig), x)
end

function Gen.random(::LogNormal, mu::Real, sig::Real)
    rand(Distributions.LogNormal(mu, sig)) 
end

# Gen.logpdf_grad(::LogNormal, x, mu, sig) = (nothing, nothing, nothing)
function Gen.logpdf_grad(::LogNormal, x::Real, mu::Real, sig::Real)
    deriv_x = Distributions.gradlogpdf(Distributions.LogNormal(mu,sig),x) #Distributions package calculates derivative with respect to x
    (_, deriv_mu, deriv_sig) = Gen.logpdf_grad(normal, log(x), mu, sig) #see derivation in notes on Jan 30 2020: it's equal to the normal distribution
    (deriv_x, deriv_mu, deriv_sig)
end

Gen.has_output_grad(::LogNormal) = true
Gen.has_argument_grads(::LogNormal) = (true,true)

"""
    TruncatedLogNormal(mu::Real,sig::Real,u::Real)
Sample an `Real` from the Log Normal with mu and sig, with upper bound = u
"""
struct TruncatedLogNormal <: Gen.Distribution{Float64} end

const truncated_log_normal = TruncatedLogNormal()

function Gen.logpdf(::TruncatedLogNormal, x::Real, mu::Real, sig::Real, u::Real)
    Distributions.logpdf(Distributions.Truncated(Distributions.LogNormal(mu, sig), 0, u), x)
end

function Gen.random(::TruncatedLogNormal, mu::Real, sig::Real, u::Real)
    rand(Distributions.Truncated(Distributions.LogNormal(mu, sig), 0, u)) 
end

function Gen.logpdf_grad(::TruncatedLogNormal, x::Real, mu::Real, sig::Real, u::Real)
    (nothing, nothing, nothing)
end

Gen.has_output_grad(::TruncatedLogNormal) = false
Gen.has_argument_grads(::TruncatedLogNormal) = (false,false)


"""
    TruncatedLogNormal(mu::Real,sig::Real,u::Real)
Sample an `Real` from the Log Normal with mu and sig, with upper bound = u
"""
struct MultivariateNormalMixture <: Gen.Distribution{Vector{Float64}} end

const mvn_mixture = MultivariateNormalMixture()

function Gen.logpdf(::MultivariateNormalMixture, x::Vector{Float64}, 
            mus::Matrix{Float64}, covs::Array{Float64}, ps::Vector{Float64})
    lp = [log(ps[cluster]) + Gen.logpdf(mvnormal, x, mus[cluster,:], covs[cluster, :, :]) for cluster in 1:length(ps)]
    logsumexp(lp)
end

function Gen.random(::MultivariateNormalMixture, mus::Matrix{Float64}, covs::Array{Float64}, ps::Vector{Float64})
    cluster = categorical(ps)
    mvnormal(mus[cluster,:], covs[cluster,:,:])
end

function Gen.logpdf_grad(::MultivariateNormalMixture, x::Vector{Float64}, 
            mus::Matrix{Float64}, covs::Array{Float64}, ps::Vector{Float64})
    (nothing, nothing, nothing, nothing)
end
is_discrete(::MultivariateNormalMixture) = false
Gen.has_output_grad(::MultivariateNormalMixture) = false
Gen.has_argument_grads(::MultivariateNormalMixture) = (false,false,false)
