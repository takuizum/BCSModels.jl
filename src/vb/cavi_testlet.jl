# Mean-field CAVI for the disjoint-testlet BCSM.
#
# Same factorisation as IRT-BCSM but with T layers (one per testlet). Under
# disjointness, the q(θ_t) updates are independent: each uses
#     β̃_t = β₀ + (1/(2 k_t²)) Σ_i [(Σ_{j∈t} (E_q[z_{ij}] + E_q[b_j]))²
#                                    + Σ_{j∈t} (Var_q[z_{ij}] + Var_q[b_j])].

"""
    cavi_testlet_bcsm(Y, model; maxiter=200, tol=1e-5, rng=Random.default_rng(),
                      verbose=false) -> VBResult

Run the mean-field coordinate-ascent variational Bayes routine for the
disjoint-testlet BCSM on an `N × K` binary response matrix `Y`.

The variational family factorises as `q(z, b, θ) = ∏_{i,j} q(z_{ij}) · ∏_j q(b_j) · ∏_t q(θ_t)`
with truncated-normal, Gaussian, and inverse-gamma marginals respectively.
The plug-in `E_q[Λ] ≈ Λ(E_q[θ])` is used in the `q(z)` and `q(b)` updates;
the `q(θ_t)` updates remain closed form. Convergence is declared when the
relative change in the joint moment vector falls below `tol`.

See [`cavi_irt_bcsm`](@ref) for the single-layer variant, `docs/theory.md`
§7 for the derivation, and `docs/mcmc_vs_vb.md` for the empirical
comparison against [`gibbs_testlet_bcsm`](@ref).
"""
function cavi_testlet_bcsm(Y::AbstractMatrix{<:Integer},
                           model::TestletBCSM;
                           maxiter::Int = 200,
                           tol::Float64 = 1e-5,
                           rng::AbstractRNG = Random.default_rng(),
                           verbose::Bool = false)
    N, K = size(Y)
    K == model.K || throw(DimensionMismatch("model.K = $(model.K), got K = $K"))
    T = n_layers(model)
    σ² = ones(K)
    U  = indicator_matrix(model)
    sizes = testlet_sizes(model)
    ψvec  = [model.ψ_scale / s for s in sizes]
    trvec = [-1.0 / s for s in sizes]
    α₀, β₀ = model.α₀, model.β₀

    m_b = zeros(K); v_b = fill(model.τ²_b, K)
    m_θ = fill(0.05, T); v_θ = fill(0.1, T)
    sign_mat = (2 .* Int.(Y) .- 1)
    m_z = float.(sign_mat) .* 0.5
    v_z = fill(1.0, N, K)

    elbo_hist = Float64[]
    converged = false
    t0 = time()
    prev_metric = Inf

    for it in 1:maxiter
        Σhat = AdditiveCovariance(σ², copy(m_θ), U)
        Λ = inv_covariance(Σhat)
        Λdiag = diag(Λ)

        # ---- q(z_{ij}) ----
        s = Λ * (m_z' .+ m_b)
        @inbounds for i in 1:N
            for j in 1:K
                v_j = 1.0 / Λdiag[j]
                m_cond = m_z[i, j] - v_j * s[j, i]
                sgn = sign_mat[i, j]
                μnew, νnew = truncnorm_moments(m_cond, v_j, sgn)
                Δm = μnew - m_z[i, j]
                for k in 1:K
                    s[k, i] += Λ[k, j] * Δm
                end
                m_z[i, j] = μnew
                v_z[i, j] = νnew - μnew^2
            end
        end

        # ---- q(b_j) ----
        v_b_new = 1.0 ./ (N .* Λdiag .+ 1 / model.τ²_b)
        m_zbar = vec(mean(m_z, dims=1))
        Λ_mbar = Λ * m_zbar
        Λ_mb = Λ * m_b
        m_b_new = similar(m_b)
        @inbounds for j in 1:K
            rhs = model.μ_b / model.τ²_b -
                  N * Λ_mbar[j] -
                  N * (Λ_mb[j] - Λdiag[j] * m_b[j])
            m_b_new[j] = v_b_new[j] * rhs
        end
        m_b = m_b_new
        v_b = v_b_new

        # ---- q(θ_t) per testlet ----
        for t in 1:T
            kt = sizes[t]
            Qexp = 0.0
            @inbounds for i in 1:N
                ms = 0.0; vs = 0.0
                for j in 1:K
                    if U[j, t] == 1.0
                        ms += m_z[i, j] + m_b[j]
                        vs += v_z[i, j] + v_b[j]
                    end
                end
                Qexp += ms^2 + vs
            end
            αtilde = α₀ + N / 2
            βtilde = β₀ + Qexp / (2 * kt^2)
            d = TruncatedShiftedInverseGamma(αtilde, βtilde, ψvec[t], trvec[t])
            new_mθ = mean_tsig(d); new_vθ = var_tsig(d)
            m_θ[t] = isnan(new_mθ) ? m_θ[t] : new_mθ
            v_θ[t] = isnan(new_vθ) ? v_θ[t] : new_vθ
        end

        # convergence on a scalar summary of parameters
        cur_metric = sum(abs2, m_b) + sum(abs2, m_θ)
        Δ = abs(cur_metric - prev_metric) / (abs(prev_metric) + 1e-12)
        prev_metric = cur_metric

        elbo = compute_elbo_testlet(model, Y, m_z, v_z, m_b, v_b, m_θ, v_θ, Λ, Λdiag)
        push!(elbo_hist, elbo)

        if verbose && (it % max(1, maxiter ÷ 10) == 0)
            @info "CAVI iter $it" Δ elbo θ̄ = mean(m_θ)
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

function compute_elbo_testlet(model::TestletBCSM, Y, m_z, v_z, m_b, v_b, m_θ, v_θ, Λ, Λdiag)
    N, K = size(m_z)
    T = n_layers(model)
    Σhat = AdditiveCovariance(ones(K), m_θ, indicator_matrix(model))
    logdetΣ = logdet(Symmetric(build_covariance(Σhat)))
    quad = 0.0
    means = m_z .+ m_b'
    for i in 1:N
        v = means[i, :]
        quad += dot(v, Λ * v)
        for j in 1:K
            quad += Λdiag[j] * (v_z[i, j] + v_b[j])
        end
    end
    log_p_z = -0.5 * (N * logdetΣ + quad)

    τ²_b = model.τ²_b; μ_b = model.μ_b
    log_p_b = -0.5 * sum(((m_b .- μ_b) .^ 2 .+ v_b) ./ τ²_b)

    sizes = testlet_sizes(model)
    log_p_θ = 0.0
    for t in 1:T
        d_prior = TruncatedShiftedInverseGamma(model.α₀, model.β₀,
                                                model.ψ_scale / sizes[t],
                                                -1.0 / sizes[t])
        log_p_θ += logpdf_tsig(d_prior, m_θ[t])
    end

    H_b = 0.5 * sum(log.(2π * ℯ .* v_b))
    H_z = 0.5 * sum(log.(2π * ℯ .* clamp.(v_z, 1e-12, Inf)))
    H_θ = 0.5 * sum(log.(2π * ℯ .* clamp.(v_θ, 1e-12, Inf)))

    return log_p_z + log_p_b + log_p_θ + H_z + H_b + H_θ
end
