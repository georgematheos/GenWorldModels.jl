struct FoldedNormal <: Gen.Distribution{Float64} end
const folded_normal = FoldedNormal()
Gen.is_discrete(::FoldedNormal) = false
Gen.has_output_grad(::FoldedNormal) = false
Gen.has_argument_grads(::FoldedNormal) = (false, false)
function Gen.random(::FoldedNormal, μ, σ)
    abs(random(Gen.normal, μ, σ))
end
function Gen.logpdf(::FoldedNormal, v, μ, σ)
    if v < 0
        -Inf
    else
        Gen.logpdf(Gen.normal, v, μ, σ) + Gen.logpdf(Gen.normal, -v, μ, σ)
    end
end
@dist shifted_folded_normal(minval, μ, σ) = minval + folded_normal(μ, σ)