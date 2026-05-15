# Model type for the IRT-BCSM (Fox 2024, JEBS Eq. 1–6).
#
# A K × K covariance Σ = I + θ · 1 1ᵀ encodes the marginal of a 1PL probit
# model after integrating out the person factor with variance θ. Item
# difficulties enter as the mean -b of the latent utilities. The single
# covariance parameter θ ≥ -1/K (Sherman–Morrison bound) governs cross-item
# correlation; θ ≤ 0 corresponds to negative average dependence and is
# admissible in BCSM (Fox & Smink 2021).

"""
    IRTBCSM(K)

Single-layer BCSM with u = 1_K. Holds priors only; data and posterior live in
the sampler return objects.
"""
Base.@kwdef struct IRTBCSM
    K::Int
    # Prior on item difficulties b
    μ_b::Float64 = 0.0
    τ²_b::Float64 = 4.0
    # Truncated shifted IG prior on θ
    α₀::Float64 = 1.0
    β₀::Float64 = 1.0
    # Shift ψ. The closed-form conditional posterior on x = θ + ψ collapses to
    # a plain inverse-gamma only when ψ exactly matches the natural BCSM choice
    # ψ = 1/c_t — for the single-layer IRT-BCSM with u = 1_K this is 1/K. With
    # any other ψ the conditional is *not* IG and the closed-form formulas
    # used in the Gibbs/CAVI updates do not apply.
    ψ::Float64 = 1.0 / K
end

n_items(m::IRTBCSM) = m.K
n_layers(::IRTBCSM) = 1

"""
    indicator_matrix(m::IRTBCSM) -> Matrix{Float64}

The single-column indicator U = 1_K used by the additive covariance struct.
"""
indicator_matrix(m::IRTBCSM) = ones(m.K, 1)

"""
    initial_state(m::IRTBCSM) -> NamedTuple

Default Markov-chain initial state.
"""
function initial_state(m::IRTBCSM)
    return (
        b = zeros(m.K),
        θ = [0.05],   # weak positive correlation
        σ² = ones(m.K),
    )
end
