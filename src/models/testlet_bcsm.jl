# Model type for the Testlet-BCSM (Fox, Wenzel & Klotzke 2021, JEBS).
#
# One rank-1 layer per testlet, with u_t a binary indicator for items in
# testlet t. Items belong to exactly one testlet. The marginal covariance is
#     Σ = I + Σ_t θ_t u_t u_tᵀ.

"""
    TestletBCSM(K, testlet_of)

`testlet_of` is a length-K vector giving the testlet membership of each item
(1-based; values 1..T). The number of testlets T = maximum(testlet_of).
"""
Base.@kwdef struct TestletBCSM
    K::Int
    testlet_of::Vector{Int}
    μ_b::Float64 = 0.0
    τ²_b::Float64 = 4.0
    α₀::Float64 = 1.0
    β₀::Float64 = 1.0
    # Shift scaling. Closed-form posterior collapses to a plain IG only when
    # ψ_t = 1/k_t (k_t = testlet size). Keep ψ_scale = 1 unless you intend
    # to swap in a Metropolis–Hastings update for θ_t.
    ψ_scale::Float64 = 1.0
end

n_items(m::TestletBCSM) = m.K
n_layers(m::TestletBCSM) = maximum(m.testlet_of)

"""
    indicator_matrix(m::TestletBCSM) -> Matrix{Float64}

K × T matrix with `U[j, t] = 1` iff item j is in testlet t.
"""
function indicator_matrix(m::TestletBCSM)
    K = m.K
    T = n_layers(m)
    U = zeros(K, T)
    @inbounds for j in 1:K
        U[j, m.testlet_of[j]] = 1.0
    end
    return U
end

"""
    testlet_sizes(m::TestletBCSM) -> Vector{Int}
"""
function testlet_sizes(m::TestletBCSM)
    T = n_layers(m)
    sizes = zeros(Int, T)
    @inbounds for j in 1:m.K
        sizes[m.testlet_of[j]] += 1
    end
    return sizes
end

"""
    shift_vector(m::TestletBCSM) -> Vector{Float64}

Per-layer shift ψ_t for the truncated shifted IG prior. Default is
`ψ_scale / size_t` so that for `ψ_scale = 2`, the shift exceeds the smallest
admissible truncation `1/size_t` and the prior stays proper.
"""
function shift_vector(m::TestletBCSM)
    sizes = testlet_sizes(m)
    return [m.ψ_scale / s for s in sizes]
end

function initial_state(m::TestletBCSM)
    T = n_layers(m)
    return (
        b = zeros(m.K),
        θ = fill(0.05, T),
        σ² = ones(m.K),
    )
end
