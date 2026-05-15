# Simulate binary responses under BCSM.
#
# Generates from the probit BCSM model directly: z_i ~ N(-b, Σ), y_{ij} = 1[z_{ij} > 0],
# with Σ built from `θ` and the indicator matrix `U`. Used by the simulation
# experiments in `scripts/sim_mcmc_vs_vb.jl`.

"""
    simulate_irt_bcsm(rng, N, K; θ_true, b_true, μ_b=0.0, σ_b=1.0) -> (Y, info)

Generate `N × K` binary responses under the single-layer IRT-BCSM. If `b_true`
is `nothing`, item difficulties are sampled from `N(μ_b, σ_b²)`.
"""
function simulate_irt_bcsm(rng::AbstractRNG, N::Int, K::Int;
                            θ_true::Float64,
                            b_true::Union{Nothing, AbstractVector}=nothing,
                            μ_b::Float64=0.0,
                            σ_b::Float64=1.0)
    b = b_true === nothing ? μ_b .+ σ_b .* randn(rng, K) : collect(float.(b_true))
    Σ = Matrix(I, K, K) .+ θ_true .* (ones(K) * ones(K)')
    L = cholesky(Symmetric(Σ)).L
    Z = (-b)' .+ randn(rng, N, K) * L'
    Y = Int.(Z .> 0)
    return Y, (b=b, θ=θ_true, K=K, N=N)
end

simulate_irt_bcsm(N::Int, K::Int; kwargs...) =
    simulate_irt_bcsm(Random.default_rng(), N, K; kwargs...)

"""
    simulate_testlet_bcsm(rng, N, K; testlet_of, θ_true, b_true, μ_b, σ_b) -> (Y, info)

Generate `N × K` binary responses under the disjoint-testlet BCSM with the
given testlet assignment and per-testlet covariance components `θ_true`.
"""
function simulate_testlet_bcsm(rng::AbstractRNG, N::Int, K::Int;
                                testlet_of::AbstractVector{<:Integer},
                                θ_true::AbstractVector{<:Real},
                                b_true::Union{Nothing, AbstractVector}=nothing,
                                μ_b::Float64=0.0,
                                σ_b::Float64=1.0)
    length(testlet_of) == K || throw(DimensionMismatch(
        "testlet_of must have length K=$K"))
    T = maximum(testlet_of)
    length(θ_true) == T || throw(DimensionMismatch(
        "θ_true must have length T=$T (got $(length(θ_true)))"))
    b = b_true === nothing ? μ_b .+ σ_b .* randn(rng, K) : collect(float.(b_true))
    U = zeros(K, T)
    for j in 1:K
        U[j, testlet_of[j]] = 1.0
    end
    Σ = Matrix(I, K, K) .+ U * Diagonal(collect(float.(θ_true))) * U'
    L = cholesky(Symmetric(Σ)).L
    Z = (-b)' .+ randn(rng, N, K) * L'
    Y = Int.(Z .> 0)
    return Y, (b=b, θ=collect(float.(θ_true)), testlet_of=collect(testlet_of), K=K, N=N)
end

simulate_testlet_bcsm(N::Int, K::Int; kwargs...) =
    simulate_testlet_bcsm(Random.default_rng(), N, K; kwargs...)
