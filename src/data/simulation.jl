# Simulate binary responses under BCSM.
#
# Generates from the probit BCSM model directly: z_i ~ N(-b, Σ), y_{ij} = 1[z_{ij} > 0],
# with Σ built from `θ` and the indicator matrix `U`. Used by the simulation
# experiments in `scripts/sim_mcmc_vs_vb.jl`.

"""
    simulate_irt_bcsm([rng], N, K; θ_true, b_true=nothing, μ_b=0.0, σ_b=1.0)
        -> (Y::Matrix{Int}, info::NamedTuple)

Generate `N × K` binary responses from the single-layer probit IRT-BCSM with
covariance `Σ = I + θ_true · 1 1ᵀ`. If `b_true` is `nothing`, item
difficulties are drawn i.i.d. from `Normal(μ_b, σ_b²)`. The returned
`info` NamedTuple carries the ground-truth `b`, `θ`, `K`, `N`.

# Examples
```julia
using BCSModels, Random
Y, info = simulate_irt_bcsm(MersenneTwister(1), 500, 10; θ_true=0.4)
size(Y)              # (500, 10)
info.θ               # 0.4
info.b               # length-10 vector of true difficulties
```
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
    simulate_testlet_bcsm([rng], N, K; testlet_of, θ_true, b_true=nothing,
                          μ_b=0.0, σ_b=1.0) -> (Y, info)

Generate `N × K` binary responses under the disjoint-testlet probit BCSM
with covariance `Σ = I + Σ_t θ_true[t] · u_t u_tᵀ`, where `u_t` is the binary
indicator of testlet `t`. `testlet_of[j]` gives the testlet that item `j`
belongs to (1-based).

# Examples
```julia
using BCSModels, Random
testlet_of = repeat(1:3, inner=4)             # 12 items, 3 testlets of size 4
θ_true = [0.3, 0.5, 0.2]
Y, info = simulate_testlet_bcsm(MersenneTwister(2), 600, 12;
                                testlet_of=testlet_of, θ_true=θ_true)
size(Y)              # (600, 12)
info.θ               # [0.3, 0.5, 0.2]
```
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
