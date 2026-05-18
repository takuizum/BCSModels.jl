# Gibbs sampler for IRT-BCSM (Fox 2024, JEBS Eqs. 1–6).
#
# Single layer with u = 1_K. Conditional updates:
#   z_{ij}  | rest  ~ TN (Albert–Chib + Geweke)
#   b       | rest  ~ N_K (closed-form conjugate)
#   θ       | rest  ~ tsIG(α₀ + N/2, β₀ + Q/(2K²), ψ, tr=-1/K)
# where Q = Σ_i (1ᵀ r_i)² with r_i = z_i + b.
#
# Identification: σ_j² = 1 (probit scale).

struct GibbsResult
    samples_b::Matrix{Float64}      # niter × K
    samples_θ::Matrix{Float64}      # niter × T
    samples_σ²::Matrix{Float64}     # niter × K (kept for compatibility; 1.0 here)
    elapsed::Float64
end

"""
    gibbs_irt_bcsm(Y, model; niter=2000, burnin=1000, thin=1, rng, init, verbose=false) -> GibbsResult

Run the IRT-BCSM Gibbs sampler on an `N × K` binary response matrix `Y` under
the single-layer model [`IRTBCSM`](@ref). Returns a [`GibbsResult`](@ref)
containing posterior samples of the item difficulties `b`, the covariance
component `θ`, and (with the probit identification) `σ² ≡ 1`.

# Arguments
- `Y`: an `N × K` integer matrix with entries in `{0, 1}`.
- `model::IRTBCSM`: model spec with prior hyperparameters.

# Keyword arguments
- `niter::Int = 2000`: number of post-burn-in samples kept.
- `burnin::Int = 1000`: number of burn-in iterations.
- `thin::Int = 1`: thinning interval.
- `rng::AbstractRNG`: random number generator.
- `init`: optional NamedTuple with fields `b`, `θ`, `σ²` to override the
  default starting state.
- `verbose::Bool = false`: print progress every 10% of iterations.

# Examples
```julia
using BCSModels, Random
rng    = MersenneTwister(1)
Y, info = simulate_irt_bcsm(rng, 500, 10; θ_true=0.4, σ_b=1.0)
model  = IRTBCSM(K=10)
res    = gibbs_irt_bcsm(Y, model; niter=1500, burnin=500)

using Statistics
mean(res.samples_θ)   # posterior mean of θ — should be near 0.4
```

See also [`cavi_irt_bcsm`](@ref) for the variational counterpart, and
`docs/theory.md` for the derivation of the conditional posteriors.
"""
function gibbs_irt_bcsm(Y::AbstractMatrix{<:Integer},
                        model::IRTBCSM;
                        niter::Int = 2000,
                        burnin::Int = 1000,
                        thin::Int = 1,
                        rng::AbstractRNG = Random.default_rng(),
                        init = nothing,
                        verbose::Bool = false)
    N, K = size(Y)
    K == model.K || throw(DimensionMismatch("model.K = $(model.K), got K = $K"))
    state = init === nothing ? initial_state(model) : init
    b   = copy(state.b)
    θ   = copy(state.θ)
    σ²  = copy(state.σ²)
    Z   = randn(rng, N, K) .* 0.5 .+ (2 .* Y .- 1) .* 0.5

    U = indicator_matrix(model)                 # K × 1
    Σstruct = AdditiveCovariance(σ², θ, U)
    Λ = inv_covariance(Σstruct)

    total_iters = burnin + niter
    n_kept = niter ÷ thin
    samples_b = Matrix{Float64}(undef, n_kept, K)
    samples_θ = Matrix{Float64}(undef, n_kept, 1)
    samples_σ² = Matrix{Float64}(undef, n_kept, K)
    kept = 0

    # prior hyperparameters
    α₀, β₀, ψ = model.α₀, model.β₀, model.ψ
    tr1 = -1.0 / K        # single-layer truncation for u = 1_K, Σ₀ = I

    t0 = time()
    for it in 1:total_iters
        # 1. latent z update
        sample_latent_z!(rng, Z, Y, b, Λ)

        # 2. b update: V^{-1} = N Λ + I/τ²_b, m = V (... )
        Vinv = Symmetric(N .* Λ .+ Diagonal(fill(1 / model.τ²_b, K)))
        V = inv(Vinv)
        # ∑_i z_i has length K
        ssum = vec(sum(Z, dims=1))
        # log p in b: -1/2 b^T V^{-1} b + b^T [μ_b/τ²_b · 1 - Λ ssum]
        rhs = (model.μ_b / model.τ²_b) .* ones(K) .- Λ * ssum
        m_b = V * rhs
        Cb = cholesky(Symmetric((V + V') / 2))
        b .= m_b .+ Cb.U' * randn(rng, K)

        # 3. θ update via tsIG.  r_i = z_i + b
        # Q = Σ_i (1^T r_i)^2 = Σ_i (Σ_j (z_{ij} + b_j))^2
        Q = 0.0
        @inbounds for i in 1:N
            s = 0.0
            for j in 1:K
                s += Z[i, j] + b[j]
            end
            Q += s * s
        end
        d = TruncatedShiftedInverseGamma(α₀ + N / 2,
                                          β₀ + Q / (2 * K^2),
                                          ψ, tr1)
        θ[1] = rand_tsig(rng, d)

        # 4. refresh Σ, Λ
        Σstruct = AdditiveCovariance(σ², θ, U)
        Λ = inv_covariance(Σstruct)

        if it > burnin && ((it - burnin) % thin == 0)
            kept += 1
            samples_b[kept, :]  .= b
            samples_θ[kept, :]  .= θ
            samples_σ²[kept, :] .= σ²
        end
        if verbose && (it % max(1, total_iters ÷ 10) == 0)
            @info "Gibbs iter $it/$total_iters" θ = θ[1] b̄ = mean(b)
        end
    end
    elapsed = time() - t0
    return GibbsResult(samples_b, samples_θ, samples_σ², elapsed)
end
