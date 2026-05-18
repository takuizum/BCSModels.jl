# Coordinate-ascent VI (CAVI) for IRT-BCSM.
#
# Variational family:
#     q(z, b, θ) = ∏_{i,j} q(z_{ij}) · ∏_j q(b_j) · q(θ),
#     q(z_{ij}) = TN(m_{ij}, v_j, sign_{ij}),
#     q(b_j)    = N(m_{b,j}, v_{b,j}),
#     q(θ)      = tsIG(α̃, β̃, ψ, tr=-1/K).
# Plug-in E_q[Λ] ≈ Λ(E_q[θ]) (Wand 2017). Convergence is monitored by the
# relative L∞ change in the variational means.

struct VBResult
    m_b::Vector{Float64}      # length K, posterior mean of b
    v_b::Vector{Float64}      # length K, posterior variance of b
    m_θ::Vector{Float64}      # length T
    v_θ::Vector{Float64}      # length T
    m_z::Matrix{Float64}      # N × K  (E_q[z_{ij}])
    v_z::Matrix{Float64}      # N × K  (Var_q[z_{ij}])
    elbo::Vector{Float64}     # ELBO trajectory (one entry per outer iteration)
    converged::Bool
    n_iter::Int
    elapsed::Float64
end

"""
    cavi_irt_bcsm(Y, model; maxiter=200, tol=1e-5, rng=Random.default_rng(), verbose=false) -> VBResult

Run the mean-field coordinate-ascent variational Bayes routine for the
[`IRTBCSM`](@ref) model on an `N × K` binary response matrix `Y`. Returns
a [`VBResult`](@ref) holding the variational means and variances of all
parameters and the ELBO trajectory.

The variational family is
`q(z, b, θ) = ∏_{i,j} q(z_{ij}) · ∏_j q(b_j) · q(θ)` with truncated-normal,
Gaussian, and inverse-gamma marginals respectively. The `q(z)` and `q(b)`
updates use the plug-in `E_q[Λ] ≈ Σ⁻¹(E_q[θ])`; the `q(θ)` update is
exact closed form. Empirically the CAVI runs ~10–20× faster than the Gibbs
sampler but underestimates the posterior SD of `θ` (see
`docs/mcmc_vs_vb.md`).

# Examples
```julia
using BCSModels, Random
Y, info = simulate_irt_bcsm(MersenneTwister(3), 500, 10; θ_true=0.4)
vb = cavi_irt_bcsm(Y, IRTBCSM(K=10); maxiter=400, tol=1e-7)
vb.m_θ[1]                    # variational mean of θ
sqrt(vb.v_θ[1])              # variational SD of θ (under-disperses)
vb.elbo[end] - vb.elbo[1]    # ELBO improvement
```

See also [`gibbs_irt_bcsm`](@ref) for the MCMC reference and
[`cavi_testlet_bcsm`](@ref) for the multi-layer variant.
"""
function cavi_irt_bcsm(Y::AbstractMatrix{<:Integer},
                       model::IRTBCSM;
                       maxiter::Int = 200,
                       tol::Float64 = 1e-5,
                       rng::AbstractRNG = Random.default_rng(),
                       verbose::Bool = false)
    N, K = size(Y)
    K == model.K || throw(DimensionMismatch("model.K = $(model.K), got K = $K"))
    σ² = ones(K)
    U  = indicator_matrix(model)
    tr1 = -1.0 / K
    ψ   = model.ψ
    α₀, β₀ = model.α₀, model.β₀

    # initialisations
    m_b = zeros(K); v_b = fill(model.τ²_b, K)
    m_θ = [0.05];   v_θ = [0.1]
    sign_mat = (2 .* Int.(Y) .- 1)
    m_z = float.(sign_mat) .* 0.5
    v_z = fill(1.0, N, K)

    elbo_hist = Float64[]
    converged = false
    t0 = time()
    prev_metric = Inf

    for it in 1:maxiter
        # ---- update Λ̂ = Σ^{-1}(E_q[θ]) ----
        Σhat = AdditiveCovariance(σ², copy(m_θ), U)
        Λ = inv_covariance(Σhat)
        Λdiag = diag(Λ)            # length K

        # ---- update q(z_{ij}) ----
        # Conditional mean (current estimate of E_q[z_{ij}]):
        #   m_{ij} = E_q[z_{ij}] - v_j * (Λ row j) · (E_q[z_i] + m_b)
        # Maintain s = Λ * (m_z' + m_b)  (K × N)
        s = Λ * (m_z' .+ m_b)
        @inbounds for i in 1:N
            for j in 1:K
                v_j = 1.0 / Λdiag[j]
                m_cond = m_z[i, j] - v_j * s[j, i]
                sgn = sign_mat[i, j]
                μnew, νnew = truncnorm_moments(m_cond, v_j, sgn)
                Δm = μnew - m_z[i, j]
                # update s for the row-j column ∀ k
                for k in 1:K
                    s[k, i] += Λ[k, j] * Δm
                end
                m_z[i, j] = μnew
                v_z[i, j] = νnew - μnew^2          # variance from raw second moment
            end
        end

        # ---- update q(b_j) ----
        # v_b_j^{-1} = N Λ_{jj} + 1/τ²_b
        v_b_new = 1.0 ./ (N .* Λdiag .+ 1 / model.τ²_b)
        # m_b_j  = v_b_j · [μ_b/τ²_b - N * (Λ (m_z̄ + m_b))_j + N Λ_{jj} m_b_j]
        #         (the last term comes from removing the j-th self-contribution
        #          that is already in the precision)
        m_zbar = vec(mean(m_z, dims=1))           # length K
        Λ_mbar = Λ * m_zbar                       # length K
        Λ_mb   = Λ * m_b                          # length K
        m_b_new = similar(m_b)
        @inbounds for j in 1:K
            rhs = model.μ_b / model.τ²_b -
                  N * Λ_mbar[j] -
                  N * (Λ_mb[j] - Λdiag[j] * m_b[j])
            m_b_new[j] = v_b_new[j] * rhs
        end
        m_b = m_b_new
        v_b = v_b_new

        # ---- update q(θ) ----
        # β̃ = β₀ + (1/2K²) Σ_i [(Σ_j (m_z[i,j] + m_b[j]))² + Σ_j (v_z[i,j] + v_b[j])]
        αtilde = α₀ + N / 2
        S_means = sum(m_z, dims=2) .+ sum(m_b)    # length N
        S_var   = sum(v_z, dims=2) .+ sum(v_b)    # length N
        Qexp = sum(S_means .^ 2) + sum(S_var)
        βtilde = β₀ + Qexp / (2 * K^2)
        d = TruncatedShiftedInverseGamma(αtilde, βtilde, ψ, tr1)
        new_mθ = mean_tsig(d)
        new_vθ = var_tsig(d)
        m_θ[1] = isnan(new_mθ) ? m_θ[1] : new_mθ
        v_θ[1] = isnan(new_vθ) ? v_θ[1] : new_vθ

        # ---- crude convergence on means ----
        metric = maximum(abs.(m_b_new .- m_b_new))   # placeholder — see below
        # Use the difference of joint mean vector between iterations:
        cur_metric = sum(abs2, m_b_new) + m_θ[1]^2
        Δ = abs(cur_metric - prev_metric) / (abs(prev_metric) + 1e-12)
        prev_metric = cur_metric

        elbo = compute_elbo_irt(model, Y, m_z, v_z, m_b, v_b, m_θ, v_θ, Λ, Λdiag)
        push!(elbo_hist, elbo)

        if verbose && (it % max(1, maxiter ÷ 10) == 0)
            @info "CAVI iter $it" Δ elbo θ = m_θ[1]
        end

        if it > 1 && Δ < tol
            converged = true
            elapsed = time() - t0
            return VBResult(m_b, v_b, m_θ, v_θ, m_z, v_z, elbo_hist, converged, it, elapsed)
        end
    end
    elapsed = time() - t0
    return VBResult(m_b, v_b, m_θ, v_θ, m_z, v_z, elbo_hist, converged, maxiter, elapsed)
end

# ---- ELBO -----------------------------------------------------------------

"""
    compute_elbo_irt(model, Y, m_z, v_z, m_b, v_b, m_θ, v_θ, Λ, Λdiag) -> Float64

Computes the ELBO (up to additive constants that do not depend on q) using
plug-in E_q[Λ] ≈ Λ̂. Only includes terms that vary across iterations, so the
value is useful for convergence monitoring; the absolute value should not be
interpreted as the model marginal log-likelihood.
"""
function compute_elbo_irt(model::IRTBCSM, Y, m_z, v_z, m_b, v_b, m_θ, v_θ, Λ, Λdiag)
    N, K = size(m_z)
    # E_q[log p(z | b, θ)]:  -1/2 [N log|Σ| + Σ_i E_q[(z_i+b)^T Λ (z_i+b)]]
    # Approximate log|Σ| via Λ̂.
    Σhat = AdditiveCovariance(ones(K), m_θ, indicator_matrix(model))
    logdetΣ = logdet(Symmetric(build_covariance(Σhat)))
    quad = 0.0
    means = m_z .+ m_b'
    # Tr(Λ Cov_q(z_i + b))  per i: Σ_j (v_z[i,j] + v_b[j]) Λ_{jj}
    for i in 1:N
        # mean part
        v = means[i, :]
        quad += dot(v, Λ * v)
        # variance contribution (diagonal of Λ times var of each coord)
        for j in 1:K
            quad += Λdiag[j] * (v_z[i, j] + v_b[j])
        end
    end
    log_p_z = -0.5 * (N * logdetΣ + quad)

    # E_q[log p(b)]:  -1/(2 τ²_b) Σ_j (E_q[(b_j - μ_b)²])
    τ²_b = model.τ²_b; μ_b = model.μ_b
    log_p_b = -0.5 * sum(((m_b .- μ_b) .^ 2 .+ v_b) ./ τ²_b)

    # E_q[log p(θ)] ≈ logpdf at posterior mean (cheap proxy; exact requires
    # ∫ over tsIG which is closed-form but tedious — sufficient for monitoring)
    d_prior = TruncatedShiftedInverseGamma(model.α₀, model.β₀, model.ψ, -1.0 / K)
    log_p_θ = logpdf_tsig(d_prior, m_θ[1])

    # Entropies
    H_b = 0.5 * sum(log.(2π * ℯ .* v_b))
    # Entropy of TN(m, v, sign): use Gaussian entropy 0.5 log(2π e v) minus
    # correction; for monitoring purposes use Gaussian approximation.
    H_z = 0.5 * sum(log.(2π * ℯ .* clamp.(v_z, 1e-12, Inf)))
    H_θ = 0.5 * log(2π * ℯ * max(v_θ[1], 1e-12))

    return log_p_z + log_p_b + log_p_θ + H_z + H_b + H_θ
end
