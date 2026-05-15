# Truncated shifted inverse-gamma distribution
#
# Density (up to a normalising constant):
#     p(θ) ∝ (θ + ψ)^{-(α+1)} exp(-β / (θ + ψ))   on  θ > tr,
# with shift ψ > -tr (so tr + ψ > 0). This is exactly the prior and full
# conditional used throughout BCSM (Klotzke & Fox 2019a, Eqs. 10–11; Fox 2024,
# Eq. 6). Setting X = θ + ψ gives X ~ InverseGamma(α, β) restricted to (L, ∞)
# where L = tr + ψ > 0, which is the form we work with internally.

"""
    TruncatedShiftedInverseGamma(α, β, ψ, tr)

A truncated shifted inverse-gamma distribution with density proportional to
`(θ + ψ)^{-(α+1)} exp(-β/(θ + ψ))` on the support `θ > tr`. The shift `ψ` must
satisfy `tr + ψ > 0`. With `ψ = 0` and `tr = 0` this reduces to the ordinary
inverse-gamma.
"""
struct TruncatedShiftedInverseGamma
    α::Float64
    β::Float64
    ψ::Float64
    tr::Float64
    function TruncatedShiftedInverseGamma(α, β, ψ, tr)
        α > 0 || throw(ArgumentError("α must be positive (got $α)"))
        β > 0 || throw(ArgumentError("β must be positive (got $β)"))
        tr + ψ ≥ 0 || throw(ArgumentError(
            "shift ψ ($ψ) must be ≥ -tr ($(-tr)); got tr+ψ = $(tr+ψ)"))
        return new(float(α), float(β), float(ψ), float(tr))
    end
end

# Internal helper: regularised lower incomplete gamma at β/L, with the L=0
# limit returning 1 (the full IG support; no truncation).
@inline _ig_lower_L(α, β, L) = L > 0 ? SpecialFunctions.gamma_inc(α, β / L, 0)[1] : 1.0

"""
    rand_tsig([rng], d::TruncatedShiftedInverseGamma) -> Float64

Sample one draw via Y = 1/X with Y ~ Gamma(α, rate=β) truncated to (0, 1/L),
L = tr + ψ. Uses inverse-CDF sampling through Distributions.jl's `truncated`.
"""
function rand_tsig(rng::AbstractRNG, d::TruncatedShiftedInverseGamma)
    L = d.tr + d.ψ
    g = Gamma(d.α, 1 / d.β)
    Y = L > 0 ? rand(rng, truncated(g; lower=0.0, upper=1 / L)) : rand(rng, g)
    return 1 / Y - d.ψ
end
rand_tsig(d::TruncatedShiftedInverseGamma) = rand_tsig(Random.default_rng(), d)

"""
    mean_tsig(d::TruncatedShiftedInverseGamma) -> Float64

E[θ] of the truncated shifted inverse-gamma. Requires α > 1.
"""
function mean_tsig(d::TruncatedShiftedInverseGamma)
    d.α > 1 || return NaN
    L = d.tr + d.ψ
    z = d.β / L
    num = _ig_lower_L(d.α - 1, d.β, L)
    den = _ig_lower_L(d.α, d.β, L)
    den > 0 || return d.tr + 0.5 * (L)  # numerical degeneracy fallback
    EX = (d.β / (d.α - 1)) * (num / den)
    return EX - d.ψ
end

"""
    var_tsig(d::TruncatedShiftedInverseGamma) -> Float64

Var(θ). Requires α > 2.
"""
function var_tsig(d::TruncatedShiftedInverseGamma)
    d.α > 2 || return NaN
    L = d.tr + d.ψ
    den = _ig_lower_L(d.α, d.β, L)
    num1 = _ig_lower_L(d.α - 1, d.β, L)
    num2 = _ig_lower_L(d.α - 2, d.β, L)
    den > 0 || return NaN
    EX = (d.β / (d.α - 1)) * (num1 / den)
    EX2 = (d.β^2 / ((d.α - 1) * (d.α - 2))) * (num2 / den)
    return EX2 - EX^2
end

"""
    mean_recip_shift(d) -> Float64

E[1/(θ + ψ)]. Closed form because 1/(θ+ψ) = Y where Y ~ Gamma(α, rate=β)
truncated to (0, 1/L). Useful for CAVI updates of the variance-component term.
"""
function mean_recip_shift(d::TruncatedShiftedInverseGamma)
    L = d.tr + d.ψ
    # E[Y | Y < 1/L] for Y ~ Gamma(α, rate=β):
    #   E[Y · 1[Y < 1/L]] / P(Y < 1/L)
    #   = (α/β) · γ_low_reg(α+1, β/L) / γ_low_reg(α, β/L)
    num = _ig_lower_L(d.α + 1, d.β, L)  # γ_low_reg(α+1, β/L)
    den = _ig_lower_L(d.α, d.β, L)
    den > 0 || return 1 / L
    return (d.α / d.β) * (num / den)
end

"""
    logpdf_tsig(d, θ) -> Float64

Log density at θ (only finite when θ > tr). Used by ELBO routines.
"""
function logpdf_tsig(d::TruncatedShiftedInverseGamma, θ::Real)
    θ > d.tr || return -Inf
    L = d.tr + d.ψ
    x = θ + d.ψ
    # log unnormalised
    logu = -(d.α + 1) * log(x) - d.β / x
    # normaliser: ∫_L^∞ x^{-(α+1)} exp(-β/x) dx = β^{-α} Γ(α) γ_low_reg(α, β/L)
    lognorm = -d.α * log(d.β) + SpecialFunctions.loggamma(d.α) +
              log(_ig_lower_L(d.α, d.β, L))
    return logu - lognorm
end
