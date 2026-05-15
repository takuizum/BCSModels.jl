# Univariate truncated-normal moments and sampling helpers.
#
# Used by the latent-utility step of the IRT-BCSM/Testlet-BCSM Gibbs sampler
# (Geweke 1991 one-at-a-time scan) and by the corresponding CAVI updates of
# `q(z_ij)`. Sign convention: `sign = +1` ⇒ truncation to (0, ∞); `sign = -1`
# ⇒ truncation to (-∞, 0). This matches the Albert–Chib augmentation used
# throughout BCSM with y_{ij} ∈ {0,1} mapped to sign = 2y - 1.

"""
    truncnorm_moments(m, v, sign) -> (μ, ν)

First two raw moments E[Z] and E[Z²] of `Z ~ N(m, v)` truncated to the half-line
indicated by `sign ∈ {+1, -1}` (positive ⇒ Z > 0). `v` is the variance.
"""
@inline function truncnorm_moments(m::Real, v::Real, sign::Int)
    σ = sqrt(v)
    α = -m / σ
    # λ = hazard ratio with sign convention
    if sign > 0
        # truncation (0, ∞)
        logΦc = StatsFuns.normlogccdf(α)   # log(1 - Φ(α))
        λ = exp(StatsFuns.normlogpdf(α) - logΦc)
        μ = m + σ * λ
        δ = λ * (λ - α)
    else
        # truncation (-∞, 0)
        logΦ = StatsFuns.normlogcdf(α)
        λ = -exp(StatsFuns.normlogpdf(α) - logΦ)
        μ = m + σ * λ
        δ = λ * (λ - α)
    end
    ν_centered = v * (1 - δ)
    ν = ν_centered + μ^2  # raw second moment
    return μ, ν
end

"""
    truncnorm_rand(rng, m, v, sign) -> Float64

Sample one draw from `N(m, v)` restricted to the half-line indicated by `sign`.
Wraps `Distributions.truncated` and avoids underflow at extreme means.
"""
@inline function truncnorm_rand(rng::AbstractRNG, m::Real, v::Real, sign::Int)
    σ = sqrt(v)
    d = Normal(m, σ)
    return sign > 0 ? rand(rng, truncated(d; lower=0.0)) :
                      rand(rng, truncated(d; upper=0.0))
end
truncnorm_rand(m, v, sign) = truncnorm_rand(Random.default_rng(), m, v, sign)
