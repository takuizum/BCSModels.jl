# Gibbs sampler for Testlet-BCSM (Fox, Wenzel & Klotzke 2021, JEBS).
#
# Each layer t has a binary indicator u_t over the items of testlet t. With
# disjoint testlets and Σ₀ = I, u_t is an eigenvector of Σ_{-t} with eigenvalue
# 1 (because u_t ⊥ u_s for s ≠ t), so Σ_{-t}^{-1} u_t = u_t and
#     c_t = u_tᵀ Σ_{-t}^{-1} u_t = k_t   (testlet size).
# The truncation bound is tr_t = -1/k_t and, with ψ_t = ψ_scale/k_t (default
# ψ_scale = 2), the conditional posterior of θ_t is
#     tsIG(α₀ + N/2,  β₀ + Q_t/(2 k_t²),  ψ_t,  tr_t),
# where Q_t = Σ_i (u_tᵀ r_i)² with r_i = z_i + b. Layers update independently.

"""
    gibbs_testlet_bcsm(Y, model; niter, burnin, thin, rng, init, verbose) -> GibbsResult

Gibbs sampler for the disjoint-testlet BCSM.
"""
function gibbs_testlet_bcsm(Y::AbstractMatrix{<:Integer},
                            model::TestletBCSM;
                            niter::Int = 2000,
                            burnin::Int = 1000,
                            thin::Int = 1,
                            rng::AbstractRNG = Random.default_rng(),
                            init = nothing,
                            verbose::Bool = false)
    N, K = size(Y)
    K == model.K || throw(DimensionMismatch("model.K = $(model.K), got K = $K"))
    T = n_layers(model)
    state = init === nothing ? initial_state(model) : init
    b  = copy(state.b)
    θ  = copy(state.θ)
    σ² = copy(state.σ²)
    Z  = randn(rng, N, K) .* 0.5 .+ (2 .* Y .- 1) .* 0.5

    U = indicator_matrix(model)            # K × T
    sizes = testlet_sizes(model)           # k_t for t = 1..T
    ψvec  = [model.ψ_scale / s for s in sizes]
    trvec = [-1.0 / s for s in sizes]

    Σstruct = AdditiveCovariance(σ², θ, U)
    Λ = inv_covariance(Σstruct)

    total_iters = burnin + niter
    n_kept = niter ÷ thin
    samples_b = Matrix{Float64}(undef, n_kept, K)
    samples_θ = Matrix{Float64}(undef, n_kept, T)
    samples_σ² = Matrix{Float64}(undef, n_kept, K)
    kept = 0

    α₀, β₀ = model.α₀, model.β₀
    t0 = time()
    for it in 1:total_iters
        sample_latent_z!(rng, Z, Y, b, Λ)

        Vinv = Symmetric(N .* Λ .+ Diagonal(fill(1 / model.τ²_b, K)))
        V = inv(Vinv)
        ssum = vec(sum(Z, dims=1))
        rhs = (model.μ_b / model.τ²_b) .* ones(K) .- Λ * ssum
        m_b = V * rhs
        Cb = cholesky(Symmetric((V + V') / 2))
        b .= m_b .+ Cb.U' * randn(rng, K)

        # θ update per layer (independent under disjoint testlets)
        # r = Z .+ b' (N × K), Q_t = Σ_i (Σ_{j∈testlet t} r_{ij})²
        R = Z .+ b'                          # N × K
        @inbounds for t in 1:T
            kt = sizes[t]
            # column-sum within testlet t
            Qt = 0.0
            for i in 1:N
                s = 0.0
                for j in 1:K
                    if U[j, t] == 1.0
                        s += R[i, j]
                    end
                end
                Qt += s * s
            end
            d = TruncatedShiftedInverseGamma(α₀ + N / 2,
                                             β₀ + Qt / (2 * kt^2),
                                             ψvec[t], trvec[t])
            θ[t] = rand_tsig(rng, d)
        end

        Σstruct = AdditiveCovariance(σ², θ, U)
        Λ = inv_covariance(Σstruct)

        if it > burnin && ((it - burnin) % thin == 0)
            kept += 1
            samples_b[kept, :]  .= b
            samples_θ[kept, :]  .= θ
            samples_σ²[kept, :] .= σ²
        end
        if verbose && (it % max(1, total_iters ÷ 10) == 0)
            @info "Gibbs iter $it/$total_iters" θ̄ = mean(θ) b̄ = mean(b)
        end
    end
    elapsed = time() - t0
    return GibbsResult(samples_b, samples_θ, samples_σ², elapsed)
end
